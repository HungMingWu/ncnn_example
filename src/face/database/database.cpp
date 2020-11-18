#include <algorithm>
#include <map>
#include <memory>
#include <detecter.h>
#include <common/common.h>
#include <orbwebai/face/database.h>
#include "stream/file_stream.h"

using namespace orbwebai::face;

class Database::Impl {
	std::string db_path_;

	bool Load(StreamReader& reader) {
		uint64_t num_faces = 0;
		const uint64_t dim_feat = orbwebai::kFaceFeatureDim;
		const uint64_t dim_name = kFaceNameDim;

		Read(reader, num_faces);

		db_.clear();
		max_index_ = -1;

		for (size_t i = 0; i < num_faces; ++i) {
			char name_arr[kFaceNameDim];
			Read(reader, name_arr, size_t(dim_name));

			std::vector<float> feat(orbwebai::kFaceFeatureDim);
			Read(reader, &feat[0], size_t(dim_feat));

			db_.insert(std::make_pair(std::string(name_arr), feat));
			max_index_ = (max_index_ > i ? max_index_ : i);
		}
		++max_index_;

		return true;
	}

	bool Save(StreamWriter& writer) const {
		const uint64_t num_faces = db_.size();
		const uint64_t dim_feat = orbwebai::kFaceFeatureDim;
		const uint64_t dim_name = kFaceNameDim;

		Write(writer, num_faces);
		for (auto& line : db_) {
			auto& name = line.first;
			auto& feat = line.second;

			char name_arr[kFaceNameDim];
			sprintf(name_arr, "%s", name.c_str());

			Write(writer, name_arr, size_t(dim_name));
			Write(writer, &feat[0], size_t(dim_feat));

		}
		return true;
	}

public:
	Impl(const std::string& db_path) : db_path_(db_path) {}
	~Impl() {}

	bool Save() const {
		std::string db_name = std::string(db_path_) + "/db";
		FileWriter ofile(db_name.c_str(), FileWriter::Binary);
		if (!ofile.is_opened()) {
			return false;
		}
		return Save(ofile);
	}

	bool Load() {
		std::string db_name = std::string(db_path_) + "/db";
		FileReader ifile(db_name.c_str(), FileWriter::Binary);
		if (!ifile.is_opened()) return false;
		return Load(ifile);
	}

	int64_t Insert(const std::string& name, const std::vector<float>& feat) {
		int64_t new_index = max_index_++;
		db_.insert(std::make_pair(name, feat));
		return new_index;
	}

	int Delete(const std::string& name) {
		std::map<std::string, std::vector<float>>::iterator it = db_.find(name);
		if (it != db_.end()) {
			db_.erase(it);
		}
		return 0;
	}

	void Clear() {
		db_.clear();
		max_index_ = 0;
	}

	float CalculateSimilarity(const std::vector<float>& feat1,
		const std::vector<float>& feat2) {
		double dot = 0;
		double norm1 = 0;
		double norm2 = 0;
		for (size_t i = 0; i < orbwebai::kFaceFeatureDim; ++i) {
			dot += feat1[i] * feat2[i];
			norm1 += feat1[i] * feat1[i];
			norm2 += feat2[i] * feat2[i];
		}

		return dot / (sqrt(norm1 * norm2) + 1e-5);
	}

	bool Compare(const std::vector<float>& feat1,
		const std::vector<float>& feat2, float* similarity) {
		if (feat1.size() == 0 || feat2.size() == 0 || !similarity) return false;
		*similarity = CalculateSimilarity(feat1, feat2);
		return true;
	}

	size_t QueryTop(const std::vector<float>& feat,
		orbwebai::query::Result* query_result) {
		std::vector<std::pair<std::string, float>> result(db_.size()); {
			size_t i = 0;
			for (auto& line : db_) {
				result[i].first = line.first;
				Compare(feat, line.second, &result[i].second);
				i++;
			}
		}

		std::partial_sort(result.begin(), result.begin() + 1, result.end(), [](
			const std::pair<std::string, float>& a, const std::pair<std::string, float>& b) -> bool {
				return a.second > b.second;
			});

		query_result->name_ = result[0].first;
		query_result->sim_ = result[0].second;

		return 0;
	}


private:
	std::map<int64_t, std::vector<float>> features_db_;
	std::map<int64_t, std::string> names_db_;
	std::map<std::string, std::vector<float>> db_;
	int64_t max_index_ = 0;
};

Database::Database(const std::string& db_path) : impl(new Database::Impl(db_path))
{
}

Database::~Database() = default;

bool Database::Save() const {
	return impl->Save();
}

bool Database::Load() {
	return impl->Load();
}

int64_t Database::Insert(const std::vector<float>& feat, const std::string& name) {
	return impl->Insert(name, feat);
}

int Database::Delete(const std::string& name) {
	return impl->Delete(name);
}

int64_t Database::QueryTop(const std::vector<float>& feat,
	orbwebai::query::Result* query_result) {
	return impl->QueryTop(feat, query_result);
}

void Database::Clear() {
	impl->Clear();
}