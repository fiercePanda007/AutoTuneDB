#define DUCKDB_EXTENSION_MAIN

#include "autoDB_extension.hpp"

#include "autoDB/src/core/autoDB.h"
#include "duckdb.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/function/pragma_function.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/main/appender.hpp"
#include "duckdb/main/connection.hpp"
#include "duckdb/main/extension_util.hpp"
#include "duckdb/main/prepared_statement.hpp"
#include "duckdb/main/query_result.hpp"
#include "utils.h"

// OpenSSL linked through vcpkg
#include <openssl/opensslv.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <map>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace duckdb {

namespace autoDBLib = autoDB;

using DoubleKey = double;
using BigIntKey = int64_t;
using UBigIntKey = uint64_t;
using IntKey = int32_t;
using Payload = double;

enum class AutoDBKeyType : uint8_t { INVALID = 0, DOUBLE, BIGINT, UBIGINT, INTEGER };

struct BenchmarkSpec {
	std::string file_name;
	std::string key_sql_type;
	AutoDBKeyType key_type;
};

struct TableSchemaInfo {
	std::vector<std::string> column_names;
	std::vector<std::string> column_types;
};

static autoDBLib::AutoDB<DoubleKey, Payload> double_autoDB_index;
static autoDBLib::AutoDB<BigIntKey, Payload> big_int_autoDB_index;
static autoDBLib::AutoDB<UBigIntKey, Payload> unsigned_big_int_autoDB_index;
static autoDBLib::AutoDB<IntKey, Payload> int_autoDB_index;

static idx_t load_end_point = 0;
static std::map<std::string, std::pair<std::string, std::string>> index_type_table_name_map;
static std::map<std::string, std::string> benchmark_name_table_map;

static std::vector<std::pair<DoubleKey, Payload>> double_index_rows;
static std::vector<std::pair<BigIntKey, Payload>> bigint_index_rows;
static std::vector<std::pair<UBigIntKey, Payload>> ubigint_index_rows;
static std::vector<std::pair<IntKey, Payload>> int_index_rows;

static std::string EscapeSqlLiteral(const std::string &value) {
	return StringUtil::Replace(value, "'", "''");
}

template <class RESULT>
static void ThrowIfError(RESULT *result, const std::string &context) {
	if (!result) {
		throw InvalidInputException(context + ": null result");
	}
	if (result->HasError()) {
		throw InvalidInputException(context + ": " + result->GetError());
	}
}

static std::string BenchmarkDirectory() {
	const char *env_dir = std::getenv("AUTODB_BENCHMARK_DIR");
	if (env_dir && env_dir[0] != '\0') {
		return std::string(env_dir);
	}
	return "src";
}

static std::string JoinPath(const std::string &dir, const std::string &file_name) {
	if (dir.empty()) {
		return file_name;
	}
	if (dir.back() == '/') {
		return dir + file_name;
	}
	return dir + "/" + file_name;
}

static bool TryGetBenchmarkSpec(const std::string &benchmark_name, BenchmarkSpec &spec) {
	auto lower_name = StringUtil::Lower(benchmark_name);
	if (lower_name == "lognormal") {
		spec.file_name = "lognormal-190M.bin.data";
		spec.key_sql_type = "BIGINT";
		spec.key_type = AutoDBKeyType::BIGINT;
		return true;
	}
	if (lower_name == "longitudes") {
		spec.file_name = "longitudes-200M.bin.data";
		spec.key_sql_type = "DOUBLE";
		spec.key_type = AutoDBKeyType::DOUBLE;
		return true;
	}
	if (lower_name == "longlat") {
		spec.file_name = "longlat-200M.bin.data";
		spec.key_sql_type = "DOUBLE";
		spec.key_type = AutoDBKeyType::DOUBLE;
		return true;
	}
	if (lower_name == "ycsb") {
		spec.file_name = "ycsb-200M.bin.data";
		spec.key_sql_type = "UBIGINT";
		spec.key_type = AutoDBKeyType::UBIGINT;
		return true;
	}
	return false;
}

static AutoDBKeyType ParseTypeString(const std::string &type_string) {
	auto normalized = StringUtil::Upper(type_string);
	if (normalized == "DOUBLE" || normalized == "FLOAT" || normalized == "REAL") {
		return AutoDBKeyType::DOUBLE;
	}
	if (normalized == "BIGINT") {
		return AutoDBKeyType::BIGINT;
	}
	if (normalized == "UBIGINT") {
		return AutoDBKeyType::UBIGINT;
	}
	if (normalized == "INTEGER" || normalized == "INT" || normalized == "INT4") {
		return AutoDBKeyType::INTEGER;
	}
	return AutoDBKeyType::INVALID;
}

static AutoDBKeyType ParseIndexTypeName(const std::string &index_type) {
	auto lower_name = StringUtil::Lower(index_type);
	if (lower_name == "double") {
		return AutoDBKeyType::DOUBLE;
	}
	if (lower_name == "bigint") {
		return AutoDBKeyType::BIGINT;
	}
	if (lower_name == "ubigint") {
		return AutoDBKeyType::UBIGINT;
	}
	if (lower_name == "int" || lower_name == "integer") {
		return AutoDBKeyType::INTEGER;
	}
	return AutoDBKeyType::INVALID;
}

static TableSchemaInfo GetTableSchemaInfo(Connection &con, const std::string &table_name) {
	auto query = "PRAGMA table_info('" + EscapeSqlLiteral(table_name) + "')";
	auto result = con.Query(query);
	ThrowIfError(result.get(), "Failed to inspect table " + table_name);

	TableSchemaInfo info;
	while (true) {
		auto chunk = result->Fetch();
		if (!chunk || chunk->size() == 0) {
			break;
		}
		for (idx_t row_idx = 0; row_idx < chunk->size(); row_idx++) {
			auto name_value = chunk->GetValue(1, row_idx).DefaultCastAs(LogicalType::VARCHAR).GetValue<std::string>();
			auto type_value = chunk->GetValue(2, row_idx).DefaultCastAs(LogicalType::VARCHAR).GetValue<std::string>();
			info.column_names.push_back(name_value);
			info.column_types.push_back(StringUtil::Upper(type_value));
		}
	}

	if (info.column_names.empty()) {
		throw InvalidInputException("Table '" + table_name + "' does not exist or has no columns");
	}
	return info;
}

static idx_t FindColumnIndex(const TableSchemaInfo &schema, const std::string &column_name) {
	auto lower_target = StringUtil::Lower(column_name);
	for (idx_t i = 0; i < schema.column_names.size(); i++) {
		if (StringUtil::Lower(schema.column_names[i]) == lower_target) {
			return i;
		}
	}
	throw InvalidInputException("Column '" + column_name + "' not found in table");
}

static std::string ResolveBenchmarkTableName(const std::string &benchmark_name) {
	auto lowered = StringUtil::Lower(benchmark_name);
	auto entry = benchmark_name_table_map.find(lowered);
	if (entry != benchmark_name_table_map.end()) {
		return entry->second;
	}
	return benchmark_name + "_benchmark";
}

template <typename K>
K CastKey(const Value &value);

template <>
DoubleKey CastKey<DoubleKey>(const Value &value) {
	return value.DefaultCastAs(LogicalType::DOUBLE).GetValue<DoubleKey>();
}

template <>
BigIntKey CastKey<BigIntKey>(const Value &value) {
	return value.DefaultCastAs(LogicalType::BIGINT).GetValue<BigIntKey>();
}

template <>
UBigIntKey CastKey<UBigIntKey>(const Value &value) {
	return value.DefaultCastAs(LogicalType::UBIGINT).GetValue<UBigIntKey>();
}

template <>
IntKey CastKey<IntKey>(const Value &value) {
	return value.DefaultCastAs(LogicalType::INTEGER).GetValue<IntKey>();
}

static Payload CastPayload(const Value &value) {
	return value.DefaultCastAs(LogicalType::DOUBLE).GetValue<Payload>();
}

template <typename K>
static std::vector<std::pair<K, Payload>> ReadKeyPayloadPairs(Connection &con, const std::string &table_name,
                                                              idx_t key_column_index) {
	auto result = con.Query("SELECT * FROM " + table_name);
	ThrowIfError(result.get(), "Failed to scan table " + table_name);

	if (result->ColumnCount() <= key_column_index + 1) {
		throw InvalidInputException("Table '" + table_name + "' must contain a payload column after the key column");
	}

	std::vector<std::pair<K, Payload>> rows;
	while (true) {
		auto chunk = result->Fetch();
		if (!chunk || chunk->size() == 0) {
			break;
		}
		for (idx_t row_idx = 0; row_idx < chunk->size(); row_idx++) {
			auto key_value = chunk->GetValue(key_column_index, row_idx);
			auto payload_value = chunk->GetValue(key_column_index + 1, row_idx);
			if (key_value.IsNull() || payload_value.IsNull()) {
				continue;
			}
			rows.emplace_back(CastKey<K>(key_value), CastPayload(payload_value));
		}
	}
	return rows;
}

template <typename K>
static void PrintIndexStats(const autoDBLib::AutoDB<K, Payload> &index) {
	auto stats = index.get_stats();
	std::cout << "Stats about the index\n";
	std::cout << "***************************\n";
	std::cout << "Number of keys : " << stats.num_keys << "\n";
	std::cout << "Number of model nodes : " << stats.num_model_nodes << "\n";
	std::cout << "Number of data nodes: " << stats.num_data_nodes << "\n";
}

template <typename K>
static void BulkLoadIndex(Connection &con, const std::string &table_name, idx_t key_column_index,
                          autoDBLib::AutoDB<K, Payload> &index, std::vector<std::pair<K, Payload>> &cache) {
	cache = ReadKeyPayloadPairs<K>(con, table_name, key_column_index);
	std::sort(cache.begin(), cache.end(), [](const std::pair<K, Payload> &lhs, const std::pair<K, Payload> &rhs) {
		return lhs.first < rhs.first;
	});

	index.clear();
	auto start_time = std::chrono::high_resolution_clock::now();
	if (!cache.empty()) {
		index.bulk_load(cache.data(), static_cast<int>(cache.size()));
	}
	auto end_time = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed_seconds = end_time - start_time;
	std::cout << "Time taken to bulk load: " << elapsed_seconds.count() << " seconds\n";
	PrintIndexStats(index);
}

template <typename K>
static void EnsureCacheForTable(Connection &con, const std::string &table_name, std::vector<std::pair<K, Payload>> &cache) {
	if (!cache.empty()) {
		return;
	}
	cache = ReadKeyPayloadPairs<K>(con, table_name, 0);
}

template <typename K>
static std::vector<K> BuildKeyVector(const std::vector<std::pair<K, Payload>> &pairs) {
	std::vector<K> keys;
	keys.reserve(pairs.size());
	for (auto &entry : pairs) {
		keys.push_back(entry.first);
	}
	return keys;
}

template <typename K>
static void RunLookupBenchmarkOneBatchAutoDB(Connection &con, const std::string &table_name, autoDBLib::AutoDB<K, Payload> &index,
                                           std::vector<std::pair<K, Payload>> &cache) {
	if (index.size() == 0) {
		std::cout << "Index is empty. Please load data into the index first.\n";
		return;
	}
	EnsureCacheForTable(con, table_name, cache);
	if (cache.empty()) {
		std::cout << "Table cache is empty. Nothing to benchmark.\n";
		return;
	}

	auto keys = BuildKeyVector(cache);
	std::random_device rd;
	std::mt19937 generator(rd());
	std::shuffle(keys.begin(), keys.end(), generator);

	Payload sum = 0;
	auto start_time = std::chrono::high_resolution_clock::now();
	for (auto &key : keys) {
		auto payload = index.get_payload(key);
		if (payload) {
			sum += *payload;
		}
	}
	auto end_time = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed_seconds = end_time - start_time;

	std::cout << "Average (autoDB): " << (sum / static_cast<double>(keys.size())) << "\n";
	std::cout << "Time taken to lookup " << keys.size() << " keys is " << elapsed_seconds.count() << " seconds\n";

	auto avg_result = con.Query("SELECT AVG(payload) FROM " + table_name + ";");
	ThrowIfError(avg_result.get(), "Failed to compute AVG(payload) from table");
	std::cout << "DuckDB AVG(payload):\n";
	avg_result->Print();
}

template <typename K>
static void RunLookupBenchmarkOneBatchART(Connection &con, const std::string &table_name,
                                          std::vector<std::pair<K, Payload>> &cache) {
	EnsureCacheForTable(con, table_name, cache);
	if (cache.empty()) {
		std::cout << "Table cache is empty. Nothing to benchmark.\n";
		return;
	}

	auto keys = BuildKeyVector(cache);
	std::random_device rd;
	std::mt19937 generator(rd());
	std::shuffle(keys.begin(), keys.end(), generator);

	auto prepared = con.Prepare("SELECT payload FROM " + table_name + " WHERE key = $1");
	ThrowIfError(prepared.get(), "Failed to prepare ART benchmark statement");

	Payload sum = 0;
	idx_t found = 0;
	auto start_time = std::chrono::high_resolution_clock::now();
	for (auto &key : keys) {
		auto result = prepared->Execute(key);
		ThrowIfError(result.get(), "Failed to execute ART benchmark lookup");

		auto chunk = result->Fetch();
		if (chunk && chunk->size() > 0) {
			sum += CastPayload(chunk->GetValue(0, 0));
			found++;
		}
	}
	auto end_time = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed_seconds = end_time - start_time;

	if (found > 0) {
		std::cout << "Average (ART): " << (sum / static_cast<double>(found)) << "\n";
	}
	std::cout << "Time taken to lookup " << keys.size() << " keys is " << elapsed_seconds.count() << " seconds\n";
}

template <typename K>
static void RunLookupWorkloadAutoDB(Connection &con, const std::string &table_name, autoDBLib::AutoDB<K, Payload> &index,
                                  std::vector<std::pair<K, Payload>> &cache) {
	if (index.size() == 0) {
		std::cout << "Index is empty. Please load data into the index first.\n";
		return;
	}
	EnsureCacheForTable(con, table_name, cache);
	if (cache.empty()) {
		std::cout << "Table cache is empty. Nothing to benchmark.\n";
		return;
	}

	auto keys = BuildKeyVector(cache);
	auto key_count = static_cast<int>(keys.size());
	if (key_count <= 0) {
		return;
	}

	const int batch_size = std::min(key_count, 100000);
	const double time_limit_minutes = 0.1;
	long long cumulative_lookups = 0;
	double cumulative_lookup_time_ns = 0;
	int batch_no = 0;

	auto workload_start_time = std::chrono::high_resolution_clock::now();
	while (true) {
		batch_no++;

		std::unique_ptr<K[]> lookup_keys(get_search_keys_zipf(keys.data(), key_count, batch_size));
		auto batch_start = std::chrono::high_resolution_clock::now();
		Payload sum = 0;
		for (int i = 0; i < batch_size; i++) {
			auto payload = index.get_payload(lookup_keys[i]);
			if (payload) {
				sum += *payload;
			}
		}
		(void)sum;
		auto batch_end = std::chrono::high_resolution_clock::now();

		auto batch_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(batch_end - batch_start).count();
		cumulative_lookup_time_ns += static_cast<double>(batch_time_ns);
		cumulative_lookups += batch_size;

		std::cout << "Batch " << batch_no << ", cumulative ops: " << cumulative_lookups
		          << "\n\tbatch throughput:\t" << (batch_size / static_cast<double>(batch_time_ns)) * 1e9
		          << " lookups/sec,\t" << (cumulative_lookups / cumulative_lookup_time_ns) * 1e9 << " lookups/sec\n";

		auto workload_elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
		                               std::chrono::high_resolution_clock::now() - workload_start_time)
		                               .count();
		if (workload_elapsed_ns > time_limit_minutes * 60.0 * 1e9) {
			break;
		}
	}

	std::cout << "Cumulative lookups: " << cumulative_lookups << "\n";
	std::cout << "Throughput : " << (cumulative_lookups / cumulative_lookup_time_ns) * 1e9 << " ops/sec\n";
}

template <typename K>
static void RunLookupWorkloadART(Connection &con, const std::string &table_name,
                                 std::vector<std::pair<K, Payload>> &cache) {
	EnsureCacheForTable(con, table_name, cache);
	if (cache.empty()) {
		std::cout << "Table cache is empty. Nothing to benchmark.\n";
		return;
	}

	auto prepared = con.Prepare("SELECT payload FROM " + table_name + " WHERE key = $1");
	ThrowIfError(prepared.get(), "Failed to prepare ART lookup workload statement");

	auto keys = BuildKeyVector(cache);
	auto key_count = static_cast<int>(keys.size());
	if (key_count <= 0) {
		return;
	}

	const int batch_size = std::min(key_count, 20000);
	const double time_limit_minutes = 0.1;
	long long cumulative_lookups = 0;
	double cumulative_lookup_time_ns = 0;
	int batch_no = 0;

	auto workload_start_time = std::chrono::high_resolution_clock::now();
	while (true) {
		batch_no++;
		std::unique_ptr<K[]> lookup_keys(get_search_keys_zipf(keys.data(), key_count, batch_size));

		auto batch_start = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < batch_size; i++) {
			auto result = prepared->Execute(lookup_keys[i]);
			ThrowIfError(result.get(), "Failed during ART lookup workload");
			auto chunk = result->Fetch();
			(void)chunk;
		}
		auto batch_end = std::chrono::high_resolution_clock::now();

		auto batch_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(batch_end - batch_start).count();
		cumulative_lookup_time_ns += static_cast<double>(batch_time_ns);
		cumulative_lookups += batch_size;

		std::cout << "Batch " << batch_no << ", cumulative ops: " << cumulative_lookups
		          << "\n\tbatch throughput:\t" << (batch_size / static_cast<double>(batch_time_ns)) * 1e9
		          << " lookups/sec,\t" << (cumulative_lookups / cumulative_lookup_time_ns) * 1e9 << " lookups/sec\n";

		auto workload_elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
		                               std::chrono::high_resolution_clock::now() - workload_start_time)
		                               .count();
		if (workload_elapsed_ns > time_limit_minutes * 60.0 * 1e9) {
			break;
		}
	}

	std::cout << "Cumulative lookups: " << cumulative_lookups << "\n";
	std::cout << "Throughput : " << (cumulative_lookups / cumulative_lookup_time_ns) * 1e9 << " ops/sec\n";
}

template <typename K>
static idx_t LoadBenchmarkDataIntoTable(Connection &con, const std::string &table_name, const std::string &benchmark_file,
                                        idx_t num_keys) {
	if (num_keys > static_cast<idx_t>(std::numeric_limits<int>::max())) {
		throw InvalidInputException("Benchmark size is too large for loader");
	}
	std::vector<K> keys(num_keys);
	if (!load_binary_data(keys.data(), static_cast<int>(num_keys), benchmark_file)) {
		throw InvalidInputException("Failed to read benchmark file: " + benchmark_file);
	}

	std::mt19937_64 payload_generator(std::random_device {}());
	Appender appender(con, table_name);
	for (idx_t i = 0; i < num_keys; i++) {
		appender.AppendRow(keys[i], static_cast<Payload>(payload_generator()));
	}
	appender.Close();
	return num_keys;
}

template <typename K>
static void InsertIntoTableAndIndex(Connection &con, const std::string &table_name, K key, Payload payload,
                                    autoDBLib::AutoDB<K, Payload> &index, std::vector<std::pair<K, Payload>> &cache) {
	auto result = con.Query("INSERT INTO " + table_name + " VALUES (?, ?)", key, payload);
	ThrowIfError(result.get(), "Failed to insert row into " + table_name);

	if (index.size() > 0) {
		index.insert(key, payload);
		cache.emplace_back(key, payload);
	}
	load_end_point++;
}

template <typename K>
static std::vector<std::pair<K, Payload>> BuildInsertionValues(const std::string &benchmark_name, idx_t start_offset,
                                                               idx_t to_insert) {
	BenchmarkSpec spec;
	if (!TryGetBenchmarkSpec(benchmark_name, spec)) {
		throw InvalidInputException("Unsupported benchmark name: " + benchmark_name);
	}

	auto file_path = JoinPath(BenchmarkDirectory(), spec.file_name);
	auto required_keys = start_offset + to_insert;
	if (required_keys > static_cast<idx_t>(std::numeric_limits<int>::max())) {
		throw InvalidInputException("Insertion benchmark request is too large");
	}

	std::vector<K> all_keys(required_keys);
	if (!load_binary_data(all_keys.data(), static_cast<int>(required_keys), file_path)) {
		throw InvalidInputException("Failed to read benchmark file: " + file_path);
	}

	std::mt19937_64 payload_generator(std::random_device {}());
	std::vector<std::pair<K, Payload>> values;
	values.reserve(to_insert);
	for (idx_t i = 0; i < to_insert; i++) {
		values.emplace_back(all_keys[start_offset + i], static_cast<Payload>(payload_generator()));
	}
	return values;
}

template <typename K>
static void RunInsertionBenchmarkAutoDB(Connection &con, const std::string &benchmark_name, const std::string &table_name,
                                      idx_t to_insert, autoDBLib::AutoDB<K, Payload> &index,
                                      std::vector<std::pair<K, Payload>> &cache) {
	auto values = BuildInsertionValues<K>(benchmark_name, load_end_point, to_insert);

	auto start_time = std::chrono::high_resolution_clock::now();
	for (auto &entry : values) {
		InsertIntoTableAndIndex(con, table_name, entry.first, entry.second, index, cache);
	}
	auto end_time = std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> elapsed_seconds = end_time - start_time;
	std::cout << "Time taken to insert " << to_insert << " keys: " << elapsed_seconds.count() << " seconds\n";
}

template <typename K>
static void RunInsertionBenchmarkART(Connection &con, const std::string &benchmark_name, const std::string &table_name,
                                     idx_t to_insert) {
	auto values = BuildInsertionValues<K>(benchmark_name, load_end_point, to_insert);

	auto start_time = std::chrono::high_resolution_clock::now();
	for (auto &entry : values) {
		auto result = con.Query("INSERT INTO " + table_name + " VALUES (?, ?)", entry.first, entry.second);
		ThrowIfError(result.get(), "Failed to insert row into " + table_name);
		load_end_point++;
	}
	auto end_time = std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> elapsed_seconds = end_time - start_time;
	std::cout << "Time taken to insert " << to_insert << " keys: " << elapsed_seconds.count() << " seconds\n";
}

inline void AutoDBScalarFun(DataChunk &args, ExpressionState &state, Vector &result) {
	(void)state;
	auto &name_vector = args.data[0];
	UnaryExecutor::Execute<string_t, string_t>(
	    name_vector, result, args.size(), [&](string_t name) { return StringVector::AddString(result, "autoDB " + name.GetString()); });
}

inline void AutoDBOpenSSLVersionScalarFun(DataChunk &args, ExpressionState &state, Vector &result) {
	(void)state;
	auto &name_vector = args.data[0];
	UnaryExecutor::Execute<string_t, string_t>(name_vector, result, args.size(), [&](string_t name) {
		return StringVector::AddString(result, "autoDB " + name.GetString() + ", my linked OpenSSL version is " +
		                                         OPENSSL_VERSION_TEXT);
	});
}

void functionLoadBenchmark(ClientContext &context, const FunctionParameters &parameters) {
	std::string table_name = parameters.values[0].GetValue<std::string>();
	std::string benchmark_name = parameters.values[1].GetValue<std::string>();
	auto benchmark_size = static_cast<idx_t>(parameters.values[2].GetValue<int32_t>());
	auto num_batches_insert = static_cast<idx_t>(parameters.values[3].GetValue<int32_t>());
	(void)num_batches_insert;

	BenchmarkSpec spec;
	if (!TryGetBenchmarkSpec(benchmark_name, spec)) {
		throw InvalidInputException("Unsupported benchmark name: " + benchmark_name);
	}

	auto benchmark_file = JoinPath(BenchmarkDirectory(), spec.file_name);
	Connection con(*context.db);
	auto create_query = "CREATE OR REPLACE TABLE " + table_name + "(key " + spec.key_sql_type + ", payload DOUBLE)";
	auto create_result = con.Query(create_query);
	ThrowIfError(create_result.get(), "Failed to create table " + table_name);

	switch (spec.key_type) {
	case AutoDBKeyType::DOUBLE:
		LoadBenchmarkDataIntoTable<DoubleKey>(con, table_name, benchmark_file, benchmark_size);
		break;
	case AutoDBKeyType::BIGINT:
		LoadBenchmarkDataIntoTable<BigIntKey>(con, table_name, benchmark_file, benchmark_size);
		break;
	case AutoDBKeyType::UBIGINT:
		LoadBenchmarkDataIntoTable<UBigIntKey>(con, table_name, benchmark_file, benchmark_size);
		break;
	default:
		throw InvalidInputException("Unsupported key type in benchmark loader");
	}

	load_end_point = benchmark_size;
	benchmark_name_table_map[StringUtil::Lower(benchmark_name)] = table_name;
	std::cout << "Loaded benchmark '" << benchmark_name << "' into table '" << table_name << "' with " << benchmark_size
	          << " rows\n";
}

void functionCreateARTIndex(ClientContext &context, const FunctionParameters &parameters) {
	std::string table_name = parameters.values[0].GetValue<std::string>();
	std::string column_name = parameters.values[1].GetValue<std::string>();

	Connection con(*context.db);
	auto schema = GetTableSchemaInfo(con, table_name);
	(void)FindColumnIndex(schema, column_name);

	auto create_query = "CREATE INDEX " + column_name + "_art_index ON " + table_name + "(" + column_name + ")";
	auto start_time = std::chrono::high_resolution_clock::now();
	auto create_result = con.Query(create_query);
	auto end_time = std::chrono::high_resolution_clock::now();

	ThrowIfError(create_result.get(), "Failed to create ART index");
	std::chrono::duration<double> elapsed_seconds = end_time - start_time;
	std::cout << "ART index created successfully in " << elapsed_seconds.count() << " seconds\n";
}

void createAutoDBIndexPragmaFunction(ClientContext &context, const FunctionParameters &parameters) {
	std::string table_name = parameters.values[0].GetValue<std::string>();
	std::string column_name = parameters.values[1].GetValue<std::string>();

	Connection con(*context.db);
	auto schema = GetTableSchemaInfo(con, table_name);
	auto key_column_index = FindColumnIndex(schema, column_name);
	if (schema.column_names.size() <= key_column_index + 1) {
		throw InvalidInputException("autoDB indexing expects a payload column immediately after the key column");
	}

	auto key_type = ParseTypeString(schema.column_types[key_column_index]);
	switch (key_type) {
	case AutoDBKeyType::DOUBLE:
		BulkLoadIndex<DoubleKey>(con, table_name, key_column_index, double_autoDB_index, double_index_rows);
		index_type_table_name_map["double"] = {table_name, column_name};
		load_end_point = double_index_rows.size();
		break;
	case AutoDBKeyType::BIGINT:
		BulkLoadIndex<BigIntKey>(con, table_name, key_column_index, big_int_autoDB_index, bigint_index_rows);
		index_type_table_name_map["bigint"] = {table_name, column_name};
		load_end_point = bigint_index_rows.size();
		break;
	case AutoDBKeyType::UBIGINT:
		BulkLoadIndex<UBigIntKey>(con, table_name, key_column_index, unsigned_big_int_autoDB_index, ubigint_index_rows);
		index_type_table_name_map["ubigint"] = {table_name, column_name};
		load_end_point = ubigint_index_rows.size();
		break;
	case AutoDBKeyType::INTEGER:
		BulkLoadIndex<IntKey>(con, table_name, key_column_index, int_autoDB_index, int_index_rows);
		index_type_table_name_map["int"] = {table_name, column_name};
		load_end_point = int_index_rows.size();
		break;
	default:
		throw InvalidInputException("Unsupported key type for autoDB index: " + schema.column_types[key_column_index]);
	}
}

void functionInsertIntoTable(ClientContext &context, const FunctionParameters &parameters) {
	std::string table_name = parameters.values[0].GetValue<std::string>();
	std::string key_type_name = parameters.values[1].GetValue<std::string>();
	std::string key_str = parameters.values[2].GetValue<std::string>();
	std::string value_str = parameters.values[3].GetValue<std::string>();

	Connection con(*context.db);
	auto payload = std::stod(value_str);
	switch (ParseIndexTypeName(key_type_name)) {
	case AutoDBKeyType::DOUBLE:
		InsertIntoTableAndIndex<DoubleKey>(con, table_name, std::stod(key_str), payload, double_autoDB_index, double_index_rows);
		break;
	case AutoDBKeyType::BIGINT:
		InsertIntoTableAndIndex<BigIntKey>(con, table_name, std::stoll(key_str), payload, big_int_autoDB_index, bigint_index_rows);
		break;
	case AutoDBKeyType::UBIGINT:
		InsertIntoTableAndIndex<UBigIntKey>(con, table_name, std::stoull(key_str), payload, unsigned_big_int_autoDB_index,
		                                   ubigint_index_rows);
		break;
	case AutoDBKeyType::INTEGER:
		InsertIntoTableAndIndex<IntKey>(con, table_name, std::stoi(key_str), payload, int_autoDB_index, int_index_rows);
		break;
	default:
		throw InvalidInputException("Unsupported key type for insert_into_table: " + key_type_name);
	}
}

void functionRunBenchmarkOneBatch(ClientContext &context, const FunctionParameters &parameters) {
	std::string benchmark_name = parameters.values[0].GetValue<std::string>();
	std::string index_name = parameters.values[1].GetValue<std::string>();
	std::string data_type_name = parameters.values[2].GetValue<std::string>();
	std::string table_name = ResolveBenchmarkTableName(benchmark_name);
	Connection con(*context.db);

	auto key_type = ParseIndexTypeName(data_type_name);
	auto use_autoDB = StringUtil::Lower(index_name) == "autodb";
	switch (key_type) {
	case AutoDBKeyType::DOUBLE:
		if (use_autoDB) {
			RunLookupBenchmarkOneBatchAutoDB<DoubleKey>(con, table_name, double_autoDB_index, double_index_rows);
		} else {
			RunLookupBenchmarkOneBatchART<DoubleKey>(con, table_name, double_index_rows);
		}
		break;
	case AutoDBKeyType::BIGINT:
		if (use_autoDB) {
			RunLookupBenchmarkOneBatchAutoDB<BigIntKey>(con, table_name, big_int_autoDB_index, bigint_index_rows);
		} else {
			RunLookupBenchmarkOneBatchART<BigIntKey>(con, table_name, bigint_index_rows);
		}
		break;
	case AutoDBKeyType::UBIGINT:
		if (use_autoDB) {
			RunLookupBenchmarkOneBatchAutoDB<UBigIntKey>(con, table_name, unsigned_big_int_autoDB_index, ubigint_index_rows);
		} else {
			RunLookupBenchmarkOneBatchART<UBigIntKey>(con, table_name, ubigint_index_rows);
		}
		break;
	case AutoDBKeyType::INTEGER:
		if (use_autoDB) {
			RunLookupBenchmarkOneBatchAutoDB<IntKey>(con, table_name, int_autoDB_index, int_index_rows);
		} else {
			RunLookupBenchmarkOneBatchART<IntKey>(con, table_name, int_index_rows);
		}
		break;
	default:
		throw InvalidInputException("Unsupported data type for run_benchmark_one_batch: " + data_type_name);
	}
}

void functionRunLookupBenchmark(ClientContext &context, const FunctionParameters &parameters) {
	std::string benchmark_name = parameters.values[0].GetValue<std::string>();
	std::string index_name = parameters.values[1].GetValue<std::string>();
	std::string table_name = ResolveBenchmarkTableName(benchmark_name);

	Connection con(*context.db);
	auto schema = GetTableSchemaInfo(con, table_name);
	if (schema.column_names.empty()) {
		throw InvalidInputException("Table '" + table_name + "' has no columns");
	}
	auto key_type = ParseTypeString(schema.column_types[0]);
	auto use_autoDB = StringUtil::Lower(index_name) == "autodb";

	switch (key_type) {
	case AutoDBKeyType::DOUBLE:
		if (use_autoDB) {
			RunLookupWorkloadAutoDB<DoubleKey>(con, table_name, double_autoDB_index, double_index_rows);
		} else {
			RunLookupWorkloadART<DoubleKey>(con, table_name, double_index_rows);
		}
		break;
	case AutoDBKeyType::BIGINT:
		if (use_autoDB) {
			RunLookupWorkloadAutoDB<BigIntKey>(con, table_name, big_int_autoDB_index, bigint_index_rows);
		} else {
			RunLookupWorkloadART<BigIntKey>(con, table_name, bigint_index_rows);
		}
		break;
	case AutoDBKeyType::UBIGINT:
		if (use_autoDB) {
			RunLookupWorkloadAutoDB<UBigIntKey>(con, table_name, unsigned_big_int_autoDB_index, ubigint_index_rows);
		} else {
			RunLookupWorkloadART<UBigIntKey>(con, table_name, ubigint_index_rows);
		}
		break;
	case AutoDBKeyType::INTEGER:
		if (use_autoDB) {
			RunLookupWorkloadAutoDB<IntKey>(con, table_name, int_autoDB_index, int_index_rows);
		} else {
			RunLookupWorkloadART<IntKey>(con, table_name, int_index_rows);
		}
		break;
	default:
		throw InvalidInputException("Unsupported key type for run_lookup_benchmark: " + schema.column_types[0]);
	}
}

void functionRunInsertionBenchmark(ClientContext &context, const FunctionParameters &parameters) {
	std::string benchmark_name = parameters.values[0].GetValue<std::string>();
	std::string data_type_name = parameters.values[1].GetValue<std::string>();
	std::string index_name = parameters.values[2].GetValue<std::string>();
	auto to_insert = static_cast<idx_t>(parameters.values[3].GetValue<int32_t>());
	std::string table_name = ResolveBenchmarkTableName(benchmark_name);

	Connection con(*context.db);
	auto key_type = ParseIndexTypeName(data_type_name);
	auto use_autoDB = StringUtil::Lower(index_name) == "autodb";
	switch (key_type) {
	case AutoDBKeyType::DOUBLE:
		if (use_autoDB) {
			RunInsertionBenchmarkAutoDB<DoubleKey>(con, benchmark_name, table_name, to_insert, double_autoDB_index, double_index_rows);
		} else {
			RunInsertionBenchmarkART<DoubleKey>(con, benchmark_name, table_name, to_insert);
		}
		break;
	case AutoDBKeyType::BIGINT:
		if (use_autoDB) {
			RunInsertionBenchmarkAutoDB<BigIntKey>(con, benchmark_name, table_name, to_insert, big_int_autoDB_index, bigint_index_rows);
		} else {
			RunInsertionBenchmarkART<BigIntKey>(con, benchmark_name, table_name, to_insert);
		}
		break;
	case AutoDBKeyType::UBIGINT:
		if (use_autoDB) {
			RunInsertionBenchmarkAutoDB<UBigIntKey>(con, benchmark_name, table_name, to_insert, unsigned_big_int_autoDB_index,
			                                     ubigint_index_rows);
		} else {
			RunInsertionBenchmarkART<UBigIntKey>(con, benchmark_name, table_name, to_insert);
		}
		break;
	case AutoDBKeyType::INTEGER:
		if (use_autoDB) {
			RunInsertionBenchmarkAutoDB<IntKey>(con, benchmark_name, table_name, to_insert, int_autoDB_index, int_index_rows);
		} else {
			RunInsertionBenchmarkART<IntKey>(con, benchmark_name, table_name, to_insert);
		}
		break;
	default:
		throw InvalidInputException("Unsupported data type for run_insertion_benchmark: " + data_type_name);
	}
}

void functionAutoDBFind(ClientContext &context, const FunctionParameters &parameters) {
	(void)context;
	std::string index_type_name = parameters.values[0].GetValue<std::string>();
	std::string key_string = parameters.values[1].GetValue<std::string>();
	auto index_type = ParseIndexTypeName(index_type_name);

	auto time_start = std::chrono::high_resolution_clock::now();
	switch (index_type) {
	case AutoDBKeyType::DOUBLE: {
		auto payload = double_autoDB_index.get_payload(std::stod(key_string));
		auto time_end = std::chrono::high_resolution_clock::now();
		if (payload) {
			std::cout << "Payload found " << *payload << "\n";
			std::chrono::duration<double> elapsed_seconds = time_end - time_start;
			std::cout << "Time taken : " << elapsed_seconds.count() << " seconds\n";
		} else {
			std::cout << "Payload not found\n";
		}
		return;
	}
	case AutoDBKeyType::BIGINT: {
		auto payload = big_int_autoDB_index.get_payload(std::stoll(key_string));
		auto time_end = std::chrono::high_resolution_clock::now();
		if (payload) {
			std::cout << "Payload found " << *payload << "\n";
			std::chrono::duration<double> elapsed_seconds = time_end - time_start;
			std::cout << "Time taken : " << elapsed_seconds.count() << " seconds\n";
		} else {
			std::cout << "Payload not found\n";
		}
		return;
	}
	case AutoDBKeyType::UBIGINT: {
		auto payload = unsigned_big_int_autoDB_index.get_payload(std::stoull(key_string));
		auto time_end = std::chrono::high_resolution_clock::now();
		if (payload) {
			std::cout << "Payload found " << *payload << "\n";
			std::chrono::duration<double> elapsed_seconds = time_end - time_start;
			std::cout << "Time taken : " << elapsed_seconds.count() << " seconds\n";
		} else {
			std::cout << "Payload not found\n";
		}
		return;
	}
	case AutoDBKeyType::INTEGER: {
		auto payload = int_autoDB_index.get_payload(std::stoi(key_string));
		auto time_end = std::chrono::high_resolution_clock::now();
		if (payload) {
			std::cout << "Payload found " << *payload << "\n";
			std::chrono::duration<double> elapsed_seconds = time_end - time_start;
			std::cout << "Time taken : " << elapsed_seconds.count() << " seconds\n";
		} else {
			std::cout << "Payload not found\n";
		}
		return;
	}
	default:
		throw InvalidInputException("Unsupported index type for autoDB_find: " + index_type_name);
	}
}

void functionAutoDBSize(ClientContext &context, const FunctionParameters &parameters) {
	(void)context;
	std::string index_type_name = parameters.values[0].GetValue<std::string>();
	auto index_type = ParseIndexTypeName(index_type_name);

	long long model_size = 0;
	long long data_size = 0;
	switch (index_type) {
	case AutoDBKeyType::DOUBLE:
		model_size = double_autoDB_index.model_size();
		data_size = double_autoDB_index.data_size();
		break;
	case AutoDBKeyType::BIGINT:
		model_size = big_int_autoDB_index.model_size();
		data_size = big_int_autoDB_index.data_size();
		break;
	case AutoDBKeyType::UBIGINT:
		model_size = unsigned_big_int_autoDB_index.model_size();
		data_size = unsigned_big_int_autoDB_index.data_size();
		break;
	case AutoDBKeyType::INTEGER:
		model_size = int_autoDB_index.model_size();
		data_size = int_autoDB_index.data_size();
		break;
	default:
		throw InvalidInputException("Unsupported index type for autoDB_size: " + index_type_name);
	}

	double model_mb = static_cast<double>(model_size) / (1024.0 * 1024.0);
	double data_mb = static_cast<double>(data_size) / (1024.0 * 1024.0);
	std::cout << "Model size " << model_mb << " MB\n";
	std::cout << "Data size " << data_mb << " MB\n";
	std::cout << "Size of the indexing structure " << (model_mb + data_mb) << " MB\n";
}

void functionAuxStorage(ClientContext &context, const FunctionParameters &parameters) {
	(void)context;
	std::string index_type_name = parameters.values[0].GetValue<std::string>();
	auto index_type = ParseIndexTypeName(index_type_name);

	long long total_size = 0;
	switch (index_type) {
	case AutoDBKeyType::DOUBLE:
		total_size = static_cast<long long>(double_index_rows.size() * sizeof(std::pair<DoubleKey, Payload>));
		break;
	case AutoDBKeyType::BIGINT:
		total_size = static_cast<long long>(bigint_index_rows.size() * sizeof(std::pair<BigIntKey, Payload>));
		break;
	case AutoDBKeyType::UBIGINT:
		total_size = static_cast<long long>(ubigint_index_rows.size() * sizeof(std::pair<UBigIntKey, Payload>));
		break;
	case AutoDBKeyType::INTEGER:
		total_size = static_cast<long long>(int_index_rows.size() * sizeof(std::pair<IntKey, Payload>));
		break;
	default:
		total_size = static_cast<long long>(double_index_rows.size() * sizeof(std::pair<DoubleKey, Payload>));
		total_size += static_cast<long long>(bigint_index_rows.size() * sizeof(std::pair<BigIntKey, Payload>));
		total_size += static_cast<long long>(ubigint_index_rows.size() * sizeof(std::pair<UBigIntKey, Payload>));
		total_size += static_cast<long long>(int_index_rows.size() * sizeof(std::pair<IntKey, Payload>));
		break;
	}

	double total_size_mb = static_cast<double>(total_size) / (1024.0 * 1024.0);
	std::cout << "Auxiliary storage size " << total_size_mb << " MB\n";
}

static void LoadInternal(DatabaseInstance &instance) {
	auto autoDB_scalar_function = ScalarFunction("autoDB", {LogicalType::VARCHAR}, LogicalType::VARCHAR, AutoDBScalarFun);
	ExtensionUtil::RegisterFunction(instance, autoDB_scalar_function);

	auto autoDB_openssl_scalar_function =
	    ScalarFunction("autoDB_openssl_version", {LogicalType::VARCHAR}, LogicalType::VARCHAR, AutoDBOpenSSLVersionScalarFun);
	ExtensionUtil::RegisterFunction(instance, autoDB_openssl_scalar_function);

	auto create_autoDB_index_function = PragmaFunction::PragmaCall("create_autoDB_index", createAutoDBIndexPragmaFunction,
	                                                            {LogicalType::VARCHAR, LogicalType::VARCHAR}, {});
	ExtensionUtil::RegisterFunction(instance, create_autoDB_index_function);

	auto load_benchmark_function = PragmaFunction::PragmaCall(
	    "load_benchmark", functionLoadBenchmark,
	    {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::INTEGER, LogicalType::INTEGER}, {});
	ExtensionUtil::RegisterFunction(instance, load_benchmark_function);

	auto run_lookup_benchmark_function =
	    PragmaFunction::PragmaCall("run_lookup_benchmark", functionRunLookupBenchmark,
	                               {LogicalType::VARCHAR, LogicalType::VARCHAR}, {});
	ExtensionUtil::RegisterFunction(instance, run_lookup_benchmark_function);

	auto create_art_index_function = PragmaFunction::PragmaCall("create_art_index", functionCreateARTIndex,
	                                                           {LogicalType::VARCHAR, LogicalType::VARCHAR},
	                                                           LogicalType::INVALID);
	ExtensionUtil::RegisterFunction(instance, create_art_index_function);

	auto insert_into_table_function = PragmaFunction::PragmaCall(
	    "insert_into_table", functionInsertIntoTable,
	    {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR}, {});
	ExtensionUtil::RegisterFunction(instance, insert_into_table_function);

	auto run_benchmark_one_batch_function = PragmaFunction::PragmaCall(
	    "run_benchmark_one_batch", functionRunBenchmarkOneBatch,
	    {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR}, {});
	ExtensionUtil::RegisterFunction(instance, run_benchmark_one_batch_function);

	auto autoDB_find_function =
	    PragmaFunction::PragmaCall("autoDB_find", functionAutoDBFind, {LogicalType::VARCHAR, LogicalType::VARCHAR}, {});
	ExtensionUtil::RegisterFunction(instance, autoDB_find_function);

	auto autoDB_size_function = PragmaFunction::PragmaCall("autoDB_size", functionAutoDBSize, {LogicalType::VARCHAR}, {});
	ExtensionUtil::RegisterFunction(instance, autoDB_size_function);

	auto auxiliary_storage_function =
	    PragmaFunction::PragmaCall("auxillary_storage_size", functionAuxStorage, {LogicalType::VARCHAR}, {});
	ExtensionUtil::RegisterFunction(instance, auxiliary_storage_function);

	auto run_insertion_benchmark_function = PragmaFunction::PragmaCall(
	    "run_insertion_benchmark", functionRunInsertionBenchmark,
	    {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::INTEGER}, {});
	ExtensionUtil::RegisterFunction(instance, run_insertion_benchmark_function);
}

void AutoDBExtension::Load(DuckDB &db) {
	LoadInternal(*db.instance);
}

std::string AutoDBExtension::Name() {
	return "autoDB";
}

} // namespace duckdb

extern "C" {

DUCKDB_EXTENSION_API void autoDB_init(duckdb::DatabaseInstance &db) {
	duckdb::LoadInternal(db);
}

DUCKDB_EXTENSION_API const char *autoDB_version() {
	return duckdb::DuckDB::LibraryVersion();
}

DUCKDB_EXTENSION_API void autodb_init(duckdb::DatabaseInstance &db) {
	autoDB_init(db);
}

DUCKDB_EXTENSION_API const char *autodb_version() {
	return autoDB_version();
}
}

#ifndef DUCKDB_EXTENSION_MAIN
#error DUCKDB_EXTENSION_MAIN not defined
#endif
