#include "csv_writer.hpp"

#include <array>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <locale>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>

namespace vpd {

namespace {

bool requires_csv_quotes(
    std::string_view value
) {
    for (char character :
         value) {
        if (
            character == ',' ||
            character == '"' ||
            character == '\n' ||
            character == '\r'
        ) {
            return true;
        }
    }

    return false;
}

std::filesystem::path auxiliary_path(
    const std::filesystem::path& path,
    std::string_view suffix
) {
    std::filesystem::path result =
        path;

    result +=
        suffix;

    return result;
}

void recover_repair_file(
    const std::filesystem::path& working
) {
    const std::filesystem::path repair =
        auxiliary_path(
            working,
            ".repair"
        );

    if (
        !std::filesystem::exists(
            repair
        )
    ) {
        return;
    }

    if (
        std::filesystem::exists(
            working
        )
    ) {
        std::filesystem::remove(
            repair
        );

        return;
    }

    std::filesystem::rename(
        repair,
        working
    );
}

template <class Integer>
std::string integer_to_string(
    Integer value
) {
    std::array<char, 64U> buffer{};

    const auto result =
        std::to_chars(
            buffer.data(),
            buffer.data() +
                buffer.size(),
            value
        );

    if (
        result.ec !=
        std::errc{}
    ) {
        throw std::runtime_error(
            "Could not format an integer for CSV output."
        );
    }

    return std::string(
        buffer.data(),
        result.ptr
    );
}

void write_csv_line(
    std::ofstream& stream,
    const CsvRow& fields
) {
    for (std::size_t index = 0U;
         index < fields.size();
         ++index) {
        if (index != 0U) {
            stream.put(
                ','
            );
        }

        stream
            << csv_escape(
                   fields[index]
               );
    }

    stream.put(
        '\n'
    );

    if (!stream) {
        throw std::runtime_error(
            "Could not write a CSV row."
        );
    }
}

template <class RecordConsumer>
bool parse_csv_records(
    const std::filesystem::path& path,
    const RecordConsumer& consume_record
) {
    std::ifstream stream(
        path,
        std::ios::in |
            std::ios::binary
    );

    if (!stream.is_open()) {
        throw std::runtime_error(
            "Could not open CSV file '" +
            path.string() +
            "'."
        );
    }

    CsvRow record;
    std::string field;

    bool inside_quotes =
        false;

    bool just_closed_quote =
        false;

    bool record_started =
        false;

    const auto finish_field =
        [&]() {
            record.push_back(
                std::move(
                    field
                )
            );

            field.clear();

            just_closed_quote =
                false;
        };

    const auto finish_record =
        [&]() {
            finish_field();

            consume_record(
                record
            );

            record.clear();

            record_started =
                false;
        };

    for (;;) {
        const int next =
            stream.get();

        if (next == EOF) {
            if (stream.bad()) {
                throw std::runtime_error(
                    "Could not read CSV file '" +
                    path.string() +
                    "'."
                );
            }

            return
                inside_quotes ||
                record_started ||
                just_closed_quote ||
                !field.empty() ||
                !record.empty();
        }

        const char character =
            static_cast<char>(
                next
            );

        if (inside_quotes) {
            if (character == '"') {
                if (stream.peek() == '"') {
                    stream.get();

                    field.push_back(
                        '"'
                    );

                    continue;
                }

                inside_quotes =
                    false;

                just_closed_quote =
                    true;

                continue;
            }

            field.push_back(
                character
            );

            record_started =
                true;

            continue;
        }

        if (just_closed_quote) {
            if (character == ',') {
                finish_field();

                record_started =
                    true;

                continue;
            }

            if (
                character == '\n' ||
                character == '\r'
            ) {
                finish_record();

                if (
                    character == '\r' &&
                    stream.peek() == '\n'
                ) {
                    stream.get();
                }

                continue;
            }

            throw std::runtime_error(
                "CSV file contains characters after "
                "a closing quote in '" +
                path.string() +
                "'."
            );
        }

        if (character == '"') {
            if (!field.empty()) {
                throw std::runtime_error(
                    "CSV file contains an unexpected "
                    "quote in '" +
                    path.string() +
                    "'."
                );
            }

            inside_quotes =
                true;

            record_started =
                true;

            continue;
        }

        if (character == ',') {
            finish_field();

            record_started =
                true;

            continue;
        }

        if (
            character == '\n' ||
            character == '\r'
        ) {
            finish_record();

            if (
                character == '\r' &&
                stream.peek() == '\n'
            ) {
                stream.get();
            }

            continue;
        }

        field.push_back(
            character
        );

        record_started =
            true;
    }
}

}  // namespace

std::string csv_escape(
    std::string_view value
) {
    if (
        !requires_csv_quotes(
            value
        )
    ) {
        return std::string(
            value
        );
    }

    std::string escaped;

    escaped.reserve(
        value.size() +
        2U
    );

    escaped.push_back(
        '"'
    );

    for (char character :
         value) {
        if (character == '"') {
            escaped.push_back(
                '"'
            );
        }

        escaped.push_back(
            character
        );
    }

    escaped.push_back(
        '"'
    );

    return escaped;
}

std::string csv_integer(
    std::int64_t value
) {
    return integer_to_string(
        value
    );
}

std::string csv_unsigned_integer(
    std::uint64_t value
) {
    return integer_to_string(
        value
    );
}

std::string csv_size(
    std::size_t value
) {
    return integer_to_string(
        value
    );
}

std::string csv_real(
    double value
) {
    if (
        !std::isfinite(
            value
        )
    ) {
        throw std::invalid_argument(
            "CSV floating-point output requires "
            "a finite value."
        );
    }

    if (value == 0.0) {
        value =
            0.0;
    }

    std::array<char, 128U> buffer{};

    const auto result =
        std::to_chars(
            buffer.data(),
            buffer.data() +
                buffer.size(),
            value,
            std::chars_format::general,
            std::numeric_limits<
                double
            >::max_digits10
        );

    if (
        result.ec !=
        std::errc{}
    ) {
        throw std::runtime_error(
            "Could not format a floating-point value "
            "for CSV output."
        );
    }

    return std::string(
        buffer.data(),
        result.ptr
    );
}

std::string csv_boolean(
    bool value
) {
    return value
        ? "true"
        : "false";
}

CsvScanResult scan_csv(
    const std::filesystem::path& path,
    const CsvRowConsumer& consume_row
) {
    CsvScanResult result;

    if (
        !std::filesystem::exists(
            path
        )
    ) {
        throw std::runtime_error(
            "CSV file does not exist: '" +
            path.string() +
            "'."
        );
    }

    if (
        std::filesystem::file_size(
            path
        ) ==
        0U
    ) {
        return result;
    }

    bool header_seen =
        false;

    result.trailing_record_incomplete =
        parse_csv_records(
            path,
            [&](const CsvRow& record) {
                if (!header_seen) {
                    result.header =
                        record;

                    header_seen =
                        true;

                    return;
                }

                if (
                    record.size() !=
                    result.header.size()
                ) {
                    throw std::runtime_error(
                        "CSV file '" +
                        path.string() +
                        "' contains a record with an "
                        "unexpected number of fields."
                    );
                }

                ++result.row_count;

                if (consume_row) {
                    consume_row(
                        record
                    );
                }
            }
        );

    return result;
}

void rewrite_csv_temporary_filtered(
    const std::filesystem::path& final_path,
    const std::vector<std::string>& header,
    const CsvRowPredicate& retain_row
) {
    if (!retain_row) {
        throw std::invalid_argument(
            "A filtered CSV rewrite requires a valid "
            "row predicate."
        );
    }

    const std::filesystem::path normalized_final =
        std::filesystem::absolute(
            final_path
        ).lexically_normal();

    const std::filesystem::path working =
        auxiliary_path(
            normalized_final,
            ".tmp"
        );

    const std::filesystem::path repair =
        auxiliary_path(
            working,
            ".repair"
        );

    recover_repair_file(
        working
    );

    if (
        !std::filesystem::exists(
            working
        )
    ) {
        return;
    }

    if (
        std::filesystem::exists(
            repair
        )
    ) {
        std::filesystem::remove(
            repair
        );
    }

    std::ofstream repair_stream(
        repair,
        std::ios::out |
            std::ios::trunc |
            std::ios::binary
    );

    if (!repair_stream.is_open()) {
        throw std::runtime_error(
            "Could not create CSV repair file '" +
            repair.string() +
            "'."
        );
    }

    repair_stream.imbue(
        std::locale::classic()
    );

    try {
        write_csv_line(
            repair_stream,
            header
        );

        const CsvScanResult scan =
            scan_csv(
                working,
                [&](const CsvRow& row) {
                    if (
                        retain_row(
                            row
                        )
                    ) {
                        write_csv_line(
                            repair_stream,
                            row
                        );
                    }
                }
            );

        if (
            scan.header !=
            header
        ) {
            throw std::runtime_error(
                "CSV schema mismatch in '" +
                working.string() +
                "'."
            );
        }

        repair_stream.flush();

        if (!repair_stream) {
            throw std::runtime_error(
                "Could not flush CSV repair file '" +
                repair.string() +
                "'."
            );
        }

        repair_stream.close();

        if (repair_stream.fail()) {
            throw std::runtime_error(
                "Could not close CSV repair file '" +
                repair.string() +
                "'."
            );
        }

        std::filesystem::remove(
            working
        );

        std::filesystem::rename(
            repair,
            working
        );
    } catch (...) {
        if (repair_stream.is_open()) {
            repair_stream.close();
        }

        std::error_code error;

        std::filesystem::remove(
            repair,
            error
        );

        throw;
    }
}

CsvWriter::CsvWriter(
    std::filesystem::path path,
    std::vector<std::string> header,
    bool append
)
    : path_(
          std::filesystem::absolute(
              std::move(
                  path
              )
          ).lexically_normal()
      ),
      temporary_path_(
          auxiliary_path(
              path_,
              ".tmp"
          )
      ),
      header_(
          std::move(
              header
          )
      ) {
    recover_repair_file(
        temporary_path_
    );

    inspect_working_file(
        append
    );

    open_stream();
}

CsvWriter::~CsvWriter() {
    abandon_noexcept();
}

const std::filesystem::path&
CsvWriter::path() const {
    return path_;
}

const std::filesystem::path&
CsvWriter::temporary_path() const {
    return temporary_path_;
}

const std::vector<std::string>&
CsvWriter::header() const {
    return header_;
}

std::size_t CsvWriter::column_count() const {
    return header_.size();
}

std::uint64_t CsvWriter::row_count() const {
    return row_count_;
}

bool CsvWriter::is_open() const {
    return stream_.is_open();
}

bool CsvWriter::is_finalized() const {
    return finalized_;
}

void CsvWriter::inspect_working_file(
    bool append
) {
    if (
        std::filesystem::exists(
            path_
        )
    ) {
        throw std::runtime_error(
            "CSV output is already finalized: '" +
            path_.string() +
            "'."
        );
    }

    if (
        !std::filesystem::exists(
            temporary_path_
        )
    ) {
        row_count_ =
            0U;

        write_header_on_open_ =
            true;

        return;
    }

    if (!append) {
        throw std::runtime_error(
            "CSV working file already exists: '" +
            temporary_path_.string() +
            "'."
        );
    }

    if (
        std::filesystem::file_size(
            temporary_path_
        ) ==
        0U
    ) {
        row_count_ =
            0U;

        write_header_on_open_ =
            true;

        return;
    }

    const CsvScanResult scan =
        scan_csv(
            temporary_path_
        );

    if (
        scan.trailing_record_incomplete
    ) {
        throw std::runtime_error(
            "CSV working file '" +
            temporary_path_.string() +
            "' contains an incomplete trailing record. "
            "Task-level resume recovery must repair "
            "the file before opening a writer."
        );
    }

    if (
        scan.header !=
        header_
    ) {
        throw std::runtime_error(
            "The existing CSV header does not match "
            "the requested schema for '" +
            temporary_path_.string() +
            "'."
        );
    }

    row_count_ =
        scan.row_count;

    write_header_on_open_ =
        false;
}

void CsvWriter::open_stream() {
    if (write_header_on_open_) {
        stream_.open(
            temporary_path_,
            std::ios::out |
                std::ios::trunc |
                std::ios::binary
        );
    } else {
        stream_.open(
            temporary_path_,
            std::ios::out |
                std::ios::app |
                std::ios::binary
        );
    }

    if (!stream_.is_open()) {
        throw std::runtime_error(
            "Could not open CSV working file '" +
            temporary_path_.string() +
            "'."
        );
    }

    stream_.imbue(
        std::locale::classic()
    );

    if (write_header_on_open_) {
        write_csv_line(
            stream_,
            header_
        );

        stream_.flush();

        if (!stream_) {
            throw std::runtime_error(
                "Could not flush the CSV header to '" +
                temporary_path_.string() +
                "'."
            );
        }

        write_header_on_open_ =
            false;
    }
}

void CsvWriter::write_row(
    const CsvRow& fields
) {
    if (
        fields.size() !=
        header_.size()
    ) {
        throw std::invalid_argument(
            "A CSV row has " +
            std::to_string(
                fields.size()
            ) +
            " fields, but the schema requires " +
            std::to_string(
                header_.size()
            ) +
            "."
        );
    }

    write_csv_line(
        stream_,
        fields
    );

    ++row_count_;
}

void CsvWriter::flush() {
    stream_.flush();

    if (!stream_) {
        throw std::runtime_error(
            "Could not flush CSV working file '" +
            temporary_path_.string() +
            "'."
        );
    }
}

void CsvWriter::close_stream_checked() {
    if (!stream_.is_open()) {
        return;
    }

    stream_.flush();

    if (!stream_) {
        stream_.close();

        throw std::runtime_error(
            "Could not flush CSV working file '" +
            temporary_path_.string() +
            "'."
        );
    }

    stream_.close();

    if (stream_.fail()) {
        throw std::runtime_error(
            "Could not close CSV working file '" +
            temporary_path_.string() +
            "'."
        );
    }
}

void CsvWriter::finalize() {
    if (finalized_) {
        return;
    }

    close_stream_checked();

    std::filesystem::rename(
        temporary_path_,
        path_
    );

    finalized_ =
        true;
}

void CsvWriter::abandon_noexcept() noexcept {
    if (finalized_) {
        return;
    }

    if (stream_.is_open()) {
        stream_.flush();
        stream_.close();
    }

    finalized_ =
        true;
}

}  // namespace vpd