import os
import sys
import argparse
import yaml
from pathlib import Path
from typing import Dict, Any
import random
from datetime import datetime, timedelta
from tqdm import tqdm

import duckdb


class DatabaseSetup:
    """Handles DuckDB database initialization and data generation."""

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path).resolve()
        self.config = self._load_config(str(self.config_path))
        self.conn: duckdb.DuckDBPyConnection | None = None

        self.duck_cfg = self.config.get("duckdb", {})
        self.db_path = self.duck_cfg.get("path", "demo.duckdb")

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def _extension_candidates(self, configured_path: Path) -> list[Path]:
        """Build candidate extension paths in priority order."""
        project_root = Path(__file__).resolve().parent
        candidates = [
            configured_path,
            project_root / "intelligent-duck" / "build" / "release" / "extension" / "autodb" / "autoDB.duckdb_extension",
            project_root / "intelligent-duck" / "build" / "release" / "extension" / "autodb" / "autodb.duckdb_extension",
            project_root / "intelligent-duck" / "build" / "release" / "extension" / "alex" / "alex.duckdb_extension",
        ]

        try:
            duckdb_version, _ = self.conn.execute("PRAGMA version").fetchone()
            duckdb_platform = self.conn.execute("PRAGMA platform").fetchone()[0]
            repository_dir = (
                project_root
                / "intelligent-duck"
                / "build"
                / "release"
                / "repository"
                / duckdb_version
                / duckdb_platform
            )
            candidates.extend([
                repository_dir / "autoDB.duckdb_extension",
                repository_dir / "autodb.duckdb_extension",
                repository_dir / "alex.duckdb_extension",
            ])
        except Exception:
            pass

        # Keep order, remove duplicates.
        seen: set[str] = set()
        unique: list[Path] = []
        for p in candidates:
            key = str(p)
            if key not in seen:
                unique.append(p)
                seen.add(key)
        return unique

    def _load_optional_extension(self):
        """Load configured extension if available, otherwise continue safely."""
        ext_path = (self.duck_cfg.get("extension_path") or "").strip()
        if not ext_path:
            return

        require_extension = bool(self.duck_cfg.get("require_extension", False))
        ext_file = Path(ext_path).expanduser()
        if not ext_file.is_absolute():
            ext_file = (self.config_path.parent / ext_file).resolve()

        candidates = self._extension_candidates(ext_file)
        resolved_path = next((candidate for candidate in candidates if candidate.exists()), None)

        if not resolved_path:
            msg = (
                "DuckDB extension not found.\n"
                f"Configured path: {ext_file}\n"
                "Checked candidates:\n  - " + "\n  - ".join(str(path) for path in candidates)
            )
            if require_extension:
                raise FileNotFoundError(msg)
            print(f"Warning: {msg}\nContinuing without extension.")
            return

        if resolved_path != ext_file:
            print(f"Warning: configured extension not found at {ext_file}")
            print(f"Using discovered extension path: {resolved_path}")

        try:
            escaped_path = resolved_path.as_posix().replace("'", "''")
            self.conn.execute(f"LOAD '{escaped_path}';")
            print(f"Loaded DuckDB extension: {resolved_path}")
        except Exception as e:
            if require_extension:
                raise
            print(f"Warning: Could not load extension '{resolved_path}': {e}")
            print("Continuing without extension.")

    def connect_demo_db(self):
        """Connect to DuckDB file."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        connect_config: Dict[str, str] = {}
        if self.duck_cfg.get("allow_unsigned_extensions", True):
            connect_config["allow_unsigned_extensions"] = "true"

        if connect_config:
            self.conn = duckdb.connect(self.db_path, config=connect_config)
        else:
            self.conn = duckdb.connect(self.db_path)

        # Optional: threads
        threads = self.duck_cfg.get("threads")
        if threads:
            try:
                self.conn.execute(f"PRAGMA threads={int(threads)};")
            except Exception:
                pass

        # Optional: load intelligent-duck extension only when available.
        self._load_optional_extension()

    def create_database(self):
        """
        For DuckDB, "creating database" means (re)creating the DB file.
        We'll delete the existing file to start fresh.
        """
        print("Creating DuckDB database file...")
        if self.db_path != ":memory:" and Path(self.db_path).exists():
            print(f"Database file {self.db_path} already exists. Deleting and recreating...")
            try:
                self.cleanup()
            except Exception:
                pass
            Path(self.db_path).unlink()

        self.connect_demo_db()
        print(f"DuckDB database ready: {self.db_path}")

    def create_schema(self):
        """Create schema in DuckDB."""
        print("Creating schema...")

        # DuckDB-friendly schema:
        # - Use IDENTITY columns instead of SERIAL.
        # - Foreign keys are optional; you can add them later if needed.
        schema_sql = """
        CREATE TABLE regions (
            region_id BIGINT PRIMARY KEY ,
            region_name VARCHAR,
            country VARCHAR,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE suppliers (
            supplier_id BIGINT PRIMARY KEY ,
            supplier_name VARCHAR,
            region_id BIGINT,
            contact_email VARCHAR,
            phone VARCHAR,
            rating DOUBLE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE products (
            product_id BIGINT PRIMARY KEY  ,
            product_name VARCHAR,
            category VARCHAR,
            supplier_id BIGINT,
            price DOUBLE,
            stock_quantity BIGINT DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE customers (
            customer_id BIGINT PRIMARY KEY  ,
            customer_name VARCHAR,
            email VARCHAR,
            region_id BIGINT,
            loyalty_tier VARCHAR,
            registration_date DATE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE orders (
            order_id BIGINT PRIMARY KEY  ,
            customer_id BIGINT,
            order_date DATE,
            order_status VARCHAR,
            total_amount DOUBLE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE transactions (
            transaction_id BIGINT PRIMARY KEY  ,
            order_id BIGINT,
            product_id BIGINT,
            quantity BIGINT,
            unit_price DOUBLE,
            discount DOUBLE DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        -- Indexes (DuckDB supports CREATE INDEX)
        CREATE INDEX idx_products_category ON products(category);
        CREATE INDEX idx_products_supplier ON products(supplier_id);
        CREATE INDEX idx_customers_region ON customers(region_id);
        CREATE INDEX idx_orders_customer ON orders(customer_id);
        CREATE INDEX idx_orders_date ON orders(order_date);
        CREATE INDEX idx_transactions_order ON transactions(order_id);
        CREATE INDEX idx_transactions_product ON transactions(product_id);
        """

        try:
            self.conn.execute(schema_sql)
            print("Schema created successfully")
        except Exception as e:
            print(f"Error creating schema: {e}")
            raise

    def generate_data(self, scale_factor: float = 1.0):
        """
        Generate sample data.

        NOTE: DuckDB inserts are very fast, but 10M+ rows can still take time.
        Start with --scale 0.05 or 0.1 for testing.
        """
        print("\nGenerating sample data...")
        print(f"Scale factor: {scale_factor}x")

        n_regions = max(1, int(10 * scale_factor))
        n_suppliers = max(1, int(1000 * scale_factor))
        n_products = max(1, int(10000 * scale_factor))
        n_customers = max(1, int(100000 * scale_factor))
        n_orders = max(1, int(500000 * scale_factor))
        n_transactions = max(1, int(2000000 * scale_factor))

        self._generate_regions(n_regions)
        self._generate_suppliers(n_suppliers, n_regions)
        self._generate_products(n_products, n_suppliers)
        self._generate_customers(n_customers, n_regions)
        self._generate_orders(n_orders, n_customers)
        self._generate_transactions(n_transactions, n_orders, n_products)

        print("\nData generation complete!")

    def _generate_regions(self, n: int):
        """Generate region data (explicit IDs for DuckDB)."""
        print("Generating regions...")

        regions = [
            ("North America", "USA"),
            ("North America", "Canada"),
            ("Europe", "UK"),
            ("Europe", "Germany"),
            ("Europe", "France"),
            ("Asia", "Japan"),
            ("Asia", "China"),
            ("Asia", "India"),
            ("South America", "Brazil"),
            ("Oceania", "Australia"),
        ]

        data = regions[:n]
        if len(data) < n:
            data = (regions * (n // len(regions) + 1))[:n]

        # Add explicit IDs: 1..n
        rows = [(i + 1, data[i][0], data[i][1]) for i in range(n)]

        self.conn.executemany(
            "INSERT INTO regions (region_id, region_name, country) VALUES (?, ?, ?)",
            rows
        )
    def _generate_suppliers(self, n: int, n_regions: int):
        """Generate supplier data (explicit supplier_id)."""
        print("Generating suppliers...")

        batch_size = 5000
        for i in tqdm(range(0, n, batch_size)):
            data = []
            for j in range(min(batch_size, n - i)):
                supplier_id = i + j + 1
                data.append((
                    supplier_id,
                    f"Supplier {i + j}",
                    random.randint(1, n_regions),
                    f"contact{i+j}@supplier.com",
                    f"+1-555-{random.randint(1000, 9999)}",
                    round(random.uniform(1.0, 5.0), 2)
                ))

            self.conn.executemany(
                """INSERT INTO suppliers
                  (supplier_id, supplier_name, region_id, contact_email, phone, rating)
                  VALUES (?, ?, ?, ?, ?, ?)""",
                data
            )


    def _generate_products(self, n: int, n_suppliers: int):
        """Generate product data (explicit product_id)."""
        print("Generating products...")

        categories = ["Electronics", "Clothing", "Food", "Books", "Toys", "Sports", "Home", "Garden", "Auto", "Health"]
        batch_size = 5000

        for i in tqdm(range(0, n, batch_size)):
            data = []
            for j in range(min(batch_size, n - i)):
                product_id = i + j + 1
                data.append((
                    product_id,
                    f"Product {i + j}",
                    random.choice(categories),
                    random.randint(1, n_suppliers),
                    round(random.uniform(10, 1000), 2),
                    random.randint(0, 10000)
                ))

            self.conn.executemany(
                """INSERT INTO products
                  (product_id, product_name, category, supplier_id, price, stock_quantity)
                  VALUES (?, ?, ?, ?, ?, ?)""",
                data
            )


    def _generate_customers(self, n: int, n_regions: int):
        """Generate customer data (explicit customer_id)."""
        print("Generating customers...")

        tiers = ["Bronze", "Silver", "Gold", "Platinum"]
        base_date = datetime.now() - timedelta(days=365 * 3)
        batch_size = 5000

        for i in tqdm(range(0, n, batch_size)):
            data = []
            for j in range(min(batch_size, n - i)):
                customer_id = i + j + 1
                days_ago = random.randint(0, 365 * 3)
                reg_date = base_date + timedelta(days=days_ago)

                data.append((
                    customer_id,
                    f"Customer {i + j}",
                    f"customer{i+j}@email.com",
                    random.randint(1, n_regions),
                    random.choice(tiers),
                    reg_date.date()
                ))

            self.conn.executemany(
                """INSERT INTO customers
                  (customer_id, customer_name, email, region_id, loyalty_tier, registration_date)
                  VALUES (?, ?, ?, ?, ?, ?)""",
                data
            )


    def _generate_orders(self, n: int, n_customers: int):
        """Generate order data (explicit order_id)."""
        print("Generating orders...")

        statuses = ["Pending", "Processing", "Shipped", "Delivered", "Cancelled"]
        base_date = datetime.now() - timedelta(days=365)
        batch_size = 5000

        for i in tqdm(range(0, n, batch_size)):
            data = []
            for j in range(min(batch_size, n - i)):
                order_id = i + j + 1
                days_ago = random.randint(0, 365)
                order_date = base_date + timedelta(days=days_ago)

                data.append((
                    order_id,
                    random.randint(1, n_customers),
                    order_date.date(),
                    random.choice(statuses),
                    round(random.uniform(50, 5000), 2)
                ))

            self.conn.executemany(
                """INSERT INTO orders
                  (order_id, customer_id, order_date, order_status, total_amount)
                  VALUES (?, ?, ?, ?, ?)""",
                data
            )


    def _generate_transactions(self, n: int, n_orders: int, n_products: int):
        """Generate transaction data (explicit transaction_id)."""
        print("Generating transactions...")

        batch_size = 5000
        for i in tqdm(range(0, n, batch_size)):
            data = []
            for j in range(min(batch_size, n - i)):
                transaction_id = i + j + 1
                data.append((
                    transaction_id,
                    random.randint(1, n_orders),
                    random.randint(1, n_products),
                    random.randint(1, 10),
                    round(random.uniform(10, 1000), 2),
                    round(random.uniform(0, 20), 2)
                ))

            self.conn.executemany(
                """INSERT INTO transactions
                  (transaction_id, order_id, product_id, quantity, unit_price, discount)
                  VALUES (?, ?, ?, ?, ?, ?)""",
                data
            )

    def create_telemetry_db(self):
        """Create SQLite telemetry DB (unchanged)."""
        print("Creating telemetry database...")
        import sqlite3

        telemetry_path = self.config["paths"]["telemetry_db"]
        Path(telemetry_path).parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(telemetry_path)
        cursor = conn.cursor()

        cursor.executescript("""
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                phase TEXT NOT NULL,
                query_type TEXT,
                execution_time REAL,
                cpu_usage REAL,
                memory_usage REAL,
                cache_hit_rate REAL,
                rows_processed INTEGER,
                plan_cost REAL,
                success INTEGER,
                query_hash TEXT,
                plan_info TEXT
            );

            CREATE TABLE IF NOT EXISTS policy_updates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                old_version INTEGER,
                new_version INTEGER,
                improvement REAL,
                validation_score REAL,
                changes TEXT
            );

            CREATE TABLE IF NOT EXISTS safety_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                severity TEXT NOT NULL,
                event_type TEXT,
                description TEXT,
                action_taken TEXT,
                context TEXT
            );

            CREATE TABLE IF NOT EXISTS meta_learning_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                generation INTEGER,
                best_fitness REAL,
                avg_fitness REAL,
                hyperparameters TEXT,
                improvements TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON metrics(timestamp);
            CREATE INDEX IF NOT EXISTS idx_metrics_phase ON metrics(phase);
            CREATE INDEX IF NOT EXISTS idx_metrics_type ON metrics(query_type);
            CREATE INDEX IF NOT EXISTS idx_policy_timestamp ON policy_updates(timestamp);
            CREATE INDEX IF NOT EXISTS idx_safety_timestamp ON safety_events(timestamp);
        """)

        conn.commit()
        conn.close()
        print("Telemetry database created")

    def verify_setup(self):
        """Verify tables exist and contain data."""
        print("\nVerifying setup...")

        tables = ["regions", "suppliers", "products", "customers", "orders", "transactions"]
        for table in tables:
            count = self.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            print(f"  {table}: {count:,} rows")

        print("\nSetup verification complete!")
        return True

    def cleanup(self):
        """Close DB connection."""
        if self.conn:
            self.conn.close()
            self.conn = None


def main():
    parser = argparse.ArgumentParser(description="Setup DuckDB database for query optimizer demo")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--scale", type=float, default=1.0, help="Data scale factor")
    parser.add_argument("--verify", action="store_true", help="Only verify setup")

    args = parser.parse_args()

    setup = DatabaseSetup(args.config)

    try:
        if args.verify:
            setup.connect_demo_db()
            setup.verify_setup()
        else:
            print("Starting DuckDB database setup...")
            print("Tip: start with --scale 0.05 or 0.1 for faster testing")
            print()

            setup.create_database()
            setup.create_schema()
            setup.generate_data(args.scale)
            setup.create_telemetry_db()
            setup.verify_setup()

            print("\nDatabase setup complete!")
            print("You can now run the demo with: python run_demo_duckdb.py")

    except Exception as e:
        print(f"\nError during setup: {e}")
        sys.exit(1)
    finally:
        setup.cleanup()


if __name__ == "__main__":
    main()
