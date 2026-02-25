#pragma once

#include "duckdb.hpp"

namespace duckdb {

class AutoDBExtension : public Extension {
public:
	void Load(DuckDB &db) override;
	std::string Name() override;
};

using AutodbExtension = AutoDBExtension;

} // namespace duckdb
