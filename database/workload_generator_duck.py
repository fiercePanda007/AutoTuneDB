import random
from datetime import datetime, timedelta
from typing import Dict, Any, Tuple


class WorkloadGenerator:
    """Generates synthetic database workload for testing (DuckDB-friendly)."""

    def __init__(self, config: Dict[str, Any], db_manager):
        self.config = config
        self.db_manager = db_manager
        self.workload_config = config["workload"]
        self.rng = random.Random(int(self.workload_config.get("seed", 20260225)))
        self.deterministic_schedule = bool(
            self.workload_config.get("deterministic_schedule", True)
        )
        self.schedule_granularity = max(
            32, int(self.workload_config.get("schedule_granularity", 120))
        )
        self.enable_autodb_queries = bool(
            self.workload_config.get("enable_autodb_queries", False)
        )

        # Use config overrides if you have them (optional)
        self.id_max = int(self.workload_config.get("id_max", 100000))
        self.default_limit_range = self.workload_config.get("parameter_ranges", {}).get("limit_rows", [10, 1000])

        # DuckDB: be explicit about date/timestamp casting to avoid failures if order_date is VARCHAR
        # DuckDB supports DATE_TRUNC, but casting makes it robust across schemas.
        self.query_templates = {
            "select_simple": [
                "SELECT * FROM customers WHERE customer_id = {id}",
                "SELECT * FROM products WHERE product_id = {id}",
                "SELECT * FROM orders WHERE order_id = {id}",
            ],
            "join_two_tables": [
                """SELECT c.customer_name, o.order_date, o.total_amount
                   FROM customers c
                   JOIN orders o ON c.customer_id = o.customer_id
                   WHERE c.customer_id = {id}
                   LIMIT {limit}""",
                """SELECT p.product_name, s.supplier_name, p.price
                   FROM products p
                   JOIN suppliers s ON p.supplier_id = s.supplier_id
                   WHERE p.category = '{category}'
                   LIMIT {limit}""",
            ],
            "join_multiple": [
                """SELECT c.customer_name, o.order_date, p.product_name, t.quantity
                   FROM customers c
                   JOIN orders o ON c.customer_id = o.customer_id
                   JOIN transactions t ON o.order_id = t.order_id
                   JOIN products p ON t.product_id = p.product_id
                   WHERE CAST(o.order_date AS DATE) >= DATE '{date}'
                   LIMIT {limit}""",
            ],
            "aggregation": [
                """SELECT category, COUNT(*) as count, AVG(price) as avg_price
                   FROM products
                   GROUP BY category
                   ORDER BY count DESC
                   LIMIT {limit}""",
                """SELECT customer_id, SUM(total_amount) as total_spent
                   FROM orders
                   WHERE CAST(order_date AS DATE) >= DATE '{date}'
                   GROUP BY customer_id
                   ORDER BY total_spent DESC
                   LIMIT {limit}""",
            ],
            "analytical": [
                """SELECT
                       DATE_TRUNC('month', CAST(order_date AS TIMESTAMP)) as month,
                       COUNT(*) as order_count,
                       SUM(total_amount) as revenue
                   FROM orders
                   WHERE CAST(order_date AS DATE) >= DATE '{date}'
                   GROUP BY month
                   ORDER BY month""",
                """SELECT
                       r.region_name,
                       COUNT(DISTINCT c.customer_id) as customer_count,
                       SUM(o.total_amount) as total_revenue
                   FROM regions r
                   JOIN customers c ON r.region_id = c.region_id
                   JOIN orders o ON c.customer_id = o.customer_id
                   WHERE CAST(o.order_date AS DATE) >= DATE '{date}'
                   GROUP BY r.region_name
                   ORDER BY total_revenue DESC
                   LIMIT {limit}""",
            ],
        }

        if self.enable_autodb_queries:
            self.query_templates["autodb_lookup"] = [
                "PRAGMA autoDB_find('bigint', '{id}');",
            ]

        self._query_type_cycle = self._build_query_type_cycle()
        self._query_type_index = 0

    def _build_query_type_cycle(self) -> list[str]:
        """
        Build a deterministic weighted cycle of query types.

        This keeps phase-to-phase distributions stable and makes learning curves
        comparable without removing parameter variability.
        """
        distribution = self.workload_config["query_distribution"]
        candidates = [
            (query_type, float(weight))
            for query_type, weight in distribution.items()
            if query_type in self.query_templates and float(weight) > 0
        ]
        if not candidates:
            return []

        if not self.deterministic_schedule:
            return [name for name, _ in candidates]

        granularity = max(self.schedule_granularity, len(candidates))
        counts = {name: 1 for name, _ in candidates}
        remaining = granularity - len(candidates)
        if remaining > 0:
            total_weight = sum(weight for _, weight in candidates)
            if total_weight <= 0:
                total_weight = float(len(candidates))
                candidates = [(name, 1.0) for name, _ in candidates]

            exact_add = {
                name: (weight / total_weight) * remaining
                for name, weight in candidates
            }
            base_add = {name: int(value) for name, value in exact_add.items()}

            for name, value in base_add.items():
                counts[name] += value

            leftover = remaining - sum(base_add.values())
            if leftover > 0:
                order = sorted(
                    candidates,
                    key=lambda item: exact_add[item[0]] - base_add[item[0]],
                    reverse=True,
                )
                for i in range(leftover):
                    counts[order[i % len(order)][0]] += 1

        cycle = []
        for name, _ in candidates:
            cycle.extend([name] * counts[name])

        self.rng.shuffle(cycle)
        return cycle

    def generate_query(self) -> Tuple[str, str]:
        """Generate a random query based on distribution."""
        distribution = self.workload_config["query_distribution"]
        candidates = [
            (query_type, weight)
            for query_type, weight in distribution.items()
            if query_type in self.query_templates and float(weight) > 0
        ]

        if not candidates:
            available = ", ".join(sorted(self.query_templates.keys()))
            configured = ", ".join(sorted(distribution.keys()))
            raise ValueError(
                "No valid query types configured. "
                f"Configured: [{configured}] | Available: [{available}]"
            )

        if self.deterministic_schedule and self._query_type_cycle:
            query_type = self._query_type_cycle[
                self._query_type_index % len(self._query_type_cycle)
            ]
            self._query_type_index += 1
        else:
            query_types = [name for name, _ in candidates]
            weights = [weight for _, weight in candidates]
            query_type = self.rng.choices(query_types, weights=weights, k=1)[0]

        template = self.rng.choice(self.query_templates[query_type])
        params = self._generate_parameters()
        query = template.format(**params)
        return query, query_type

    def _generate_parameters(self) -> Dict[str, Any]:
        """Generate random query parameters (DuckDB-safe)."""
        param_ranges = self.workload_config.get("parameter_ranges", {})

        # Random date in the past year
        date_range = param_ranges.get("date_range_days", 365)
        if isinstance(date_range, list):
            days_ago = self.rng.randint(date_range[0], date_range[1])
        else:
            days_ago = self.rng.randint(1, int(date_range))
        date = (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d")

        # Random categories (escape single quotes to keep SQL valid)
        categories = [
            "Electronics", "Clothing", "Food", "Books", "Toys",
            "Sports", "Home", "Garden", "Auto", "Health"
        ]
        category = self.rng.choice(categories).replace("'", "''")

        limit_range = param_ranges.get("limit_rows", self.default_limit_range)
        limit_val = self.rng.randint(int(limit_range[0]), int(limit_range[1]))

        return {
            "id": self.rng.randint(1, self.id_max),
            "limit": limit_val,
            "date": date,
            "category": category,
            "price_min": self.rng.randint(1, 500),
            "price_max": self.rng.randint(500, 10000),
        }
