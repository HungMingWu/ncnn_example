#pragma once
#include <memory>
namespace orbwebai
{
	namespace face
	{
		class Database {
			class Impl;
			std::unique_ptr<Impl> impl;
		public:
			Database(const std::string& db_path);
			~Database();
			Database(const Database& other) = delete;
			const Database& operator=(const Database& other) = delete;
			bool Save() const;
			bool Load();
			int64_t Insert(const std::vector<float>& feat, const std::string& name);
			int Delete(const std::string& name);
			int64_t QueryTop(const std::vector<float>& feat, orbwebai::query::Result* query_result = nullptr);
			void Clear();
		};
	}
}