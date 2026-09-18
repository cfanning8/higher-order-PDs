#ifndef GRAPH_BENCHMARK_CSV_WRITER_HPP
#define GRAPH_BENCHMARK_CSV_WRITER_HPP

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

namespace vpd {

using CsvRow =
    std::vector<std::string>;

using CsvRowConsumer =
    std::function<void(
        const CsvRow&
    )>;

using CsvRowPredicate =
    std::function<bool(
        const CsvRow&
    )>;

struct CsvScanResult {
    std::vector<std::string> header;

    std::uint64_t row_count{};

    bool trailing_record_incomplete{};
};

std::string csv_escape(
    std::string_view value
);

std::string csv_integer(
    std::int64_t value
);

std::string csv_unsigned_integer(
    std::uint64_t value
);

std::string csv_size(
    std::size_t value
);

std::string csv_real(
    double value
);

std::string csv_boolean(
    bool value
);

CsvScanResult scan_csv(
    const std::filesystem::path& path,
    const CsvRowConsumer& consume_row = {}
);

void rewrite_csv_temporary_filtered(
    const std::filesystem::path& final_path,
    const std::vector<std::string>& header,
    const CsvRowPredicate& retain_row
);

class CsvWriter {
public:
    CsvWriter(
        std::filesystem::path path,
        std::vector<std::string> header,
        bool append
    );

    CsvWriter(
        const CsvWriter&
    ) = delete;

    CsvWriter& operator=(
        const CsvWriter&
    ) = delete;

    CsvWriter(
        CsvWriter&&
    ) = delete;

    CsvWriter& operator=(
        CsvWriter&&
    ) = delete;

    ~CsvWriter();

    const std::filesystem::path& path() const;

    const std::filesystem::path&
    temporary_path() const;

    const std::vector<std::string>&
    header() const;

    std::size_t column_count() const;

    std::uint64_t row_count() const;

    bool is_open() const;

    bool is_finalized() const;

    void write_row(
        const CsvRow& fields
    );

    void flush();

    void finalize();

private:
    std::filesystem::path path_;

    std::filesystem::path temporary_path_;

    std::vector<std::string> header_;

    std::ofstream stream_;

    std::uint64_t row_count_{};

    bool finalized_{};

    bool write_header_on_open_{};

    void inspect_working_file(
        bool append
    );

    void open_stream();

    void close_stream_checked();

    void abandon_noexcept() noexcept;
};

}  // namespace vpd

#endif