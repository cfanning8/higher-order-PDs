#include "metadata_writer.hpp"

#include "csv_writer.hpp"

#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace vpd {

namespace {

constexpr int kMetadataSchemaVersion =
    2;

constexpr std::string_view kExperimentName =
    "random_graph_classification";

enum class ExperimentOutputStatus {
    InProgress,
    Complete,
    Failed
};

struct JsonValue {
    enum class Type {
        Null,
        Boolean,
        Number,
        String,
        Array,
        Object
    };

    Type type{
        Type::Null
    };

    bool boolean{};

    std::string scalar;

    std::vector<JsonValue> array;

    std::map<std::string, JsonValue> object;
};

void append_unicode_code_point(
    std::string& output,
    std::uint32_t code_point
) {
    if (code_point <= 0x7fU) {
        output.push_back(
            static_cast<char>(
                code_point
            )
        );

        return;
    }

    if (code_point <= 0x7ffU) {
        output.push_back(
            static_cast<char>(
                0xc0U |
                (
                    code_point >>
                    6U
                )
            )
        );

        output.push_back(
            static_cast<char>(
                0x80U |
                (
                    code_point &
                    0x3fU
                )
            )
        );

        return;
    }

    if (code_point <= 0xffffU) {
        output.push_back(
            static_cast<char>(
                0xe0U |
                (
                    code_point >>
                    12U
                )
            )
        );

        output.push_back(
            static_cast<char>(
                0x80U |
                (
                    (
                        code_point >>
                        6U
                    ) &
                    0x3fU
                )
            )
        );

        output.push_back(
            static_cast<char>(
                0x80U |
                (
                    code_point &
                    0x3fU
                )
            )
        );

        return;
    }

    output.push_back(
        static_cast<char>(
            0xf0U |
            (
                code_point >>
                18U
            )
        )
    );

    output.push_back(
        static_cast<char>(
            0x80U |
            (
                (
                    code_point >>
                    12U
                ) &
                0x3fU
            )
        )
    );

    output.push_back(
        static_cast<char>(
            0x80U |
            (
                (
                    code_point >>
                    6U
                ) &
                0x3fU
            )
        )
    );

    output.push_back(
        static_cast<char>(
            0x80U |
            (
                code_point &
                0x3fU
            )
        )
    );
}

class JsonParser {
public:
    explicit JsonParser(
        std::string_view text
    )
        : text_(
              text
          ) {
    }

    JsonValue parse() {
        skip_whitespace();

        JsonValue value =
            parse_value();

        skip_whitespace();

        if (
            position_ !=
            text_.size()
        ) {
            throw std::runtime_error(
                "Metadata JSON contains trailing "
                "characters."
            );
        }

        return value;
    }

private:
    std::string_view text_;

    std::size_t position_{};

    bool at_end() const {
        return
            position_ >=
            text_.size();
    }

    char peek() const {
        if (at_end()) {
            throw std::runtime_error(
                "Metadata JSON ended unexpectedly."
            );
        }

        return text_[position_];
    }

    char take() {
        const char character =
            peek();

        ++position_;

        return character;
    }

    void skip_whitespace() {
        while (
            !at_end() &&
            (
                text_[position_] == ' ' ||
                text_[position_] == '\t' ||
                text_[position_] == '\n' ||
                text_[position_] == '\r'
            )
        ) {
            ++position_;
        }
    }

    void require_character(
        char expected
    ) {
        if (
            take() !=
            expected
        ) {
            throw std::runtime_error(
                "Metadata JSON contains invalid syntax."
            );
        }
    }

    void require_literal(
        std::string_view literal
    ) {
        if (
            literal.size() >
            text_.size() -
                position_ ||
            text_.substr(
                position_,
                literal.size()
            ) !=
                literal
        ) {
            throw std::runtime_error(
                "Metadata JSON contains an invalid "
                "literal."
            );
        }

        position_ +=
            literal.size();
    }

    static unsigned int hex_value(
        char character
    ) {
        if (
            character >= '0' &&
            character <= '9'
        ) {
            return static_cast<unsigned int>(
                character -
                '0'
            );
        }

        if (
            character >= 'a' &&
            character <= 'f'
        ) {
            return
                10U +
                static_cast<unsigned int>(
                    character -
                    'a'
                );
        }

        if (
            character >= 'A' &&
            character <= 'F'
        ) {
            return
                10U +
                static_cast<unsigned int>(
                    character -
                    'A'
                );
        }

        throw std::runtime_error(
            "Metadata JSON contains an invalid "
            "Unicode escape."
        );
    }

    unsigned int parse_unicode_code_unit() {
        if (
            text_.size() -
                position_ <
            4U
        ) {
            throw std::runtime_error(
                "Metadata JSON contains a truncated "
                "Unicode escape."
            );
        }

        unsigned int code_unit =
            0U;

        for (int index = 0;
             index < 4;
             ++index) {
            code_unit =
                (
                    code_unit <<
                    4U
                ) |
                hex_value(
                    take()
                );
        }

        return code_unit;
    }

    std::uint32_t parse_unicode_code_point() {
        const unsigned int first =
            parse_unicode_code_unit();

        if (
            first >= 0xdc00U &&
            first <= 0xdfffU
        ) {
            throw std::runtime_error(
                "Metadata JSON contains an isolated "
                "low surrogate."
            );
        }

        if (
            first < 0xd800U ||
            first > 0xdbffU
        ) {
            return first;
        }

        if (
            text_.size() -
                position_ <
                2U ||
            text_[position_] != '\\' ||
            text_[position_ + 1U] != 'u'
        ) {
            throw std::runtime_error(
                "Metadata JSON contains an isolated "
                "high surrogate."
            );
        }

        position_ +=
            2U;

        const unsigned int second =
            parse_unicode_code_unit();

        if (
            second < 0xdc00U ||
            second > 0xdfffU
        ) {
            throw std::runtime_error(
                "Metadata JSON contains an invalid "
                "surrogate pair."
            );
        }

        return
            0x10000U +
            (
                (
                    first -
                    0xd800U
                ) <<
                10U
            ) +
            (
                second -
                0xdc00U
            );
    }

    std::string parse_string() {
        require_character(
            '"'
        );

        std::string result;

        while (!at_end()) {
            const unsigned char character =
                static_cast<unsigned char>(
                    take()
                );

            if (character == '"') {
                return result;
            }

            if (character < 0x20U) {
                throw std::runtime_error(
                    "Metadata JSON contains an invalid "
                    "control character."
                );
            }

            if (character != '\\') {
                result.push_back(
                    static_cast<char>(
                        character
                    )
                );

                continue;
            }

            if (at_end()) {
                throw std::runtime_error(
                    "Metadata JSON contains a truncated "
                    "escape sequence."
                );
            }

            switch (take()) {
                case '"':
                    result.push_back(
                        '"'
                    );

                    break;

                case '\\':
                    result.push_back(
                        '\\'
                    );

                    break;

                case '/':
                    result.push_back(
                        '/'
                    );

                    break;

                case 'b':
                    result.push_back(
                        '\b'
                    );

                    break;

                case 'f':
                    result.push_back(
                        '\f'
                    );

                    break;

                case 'n':
                    result.push_back(
                        '\n'
                    );

                    break;

                case 'r':
                    result.push_back(
                        '\r'
                    );

                    break;

                case 't':
                    result.push_back(
                        '\t'
                    );

                    break;

                case 'u':
                    append_unicode_code_point(
                        result,
                        parse_unicode_code_point()
                    );

                    break;

                default:
                    throw std::runtime_error(
                        "Metadata JSON contains an "
                        "invalid escape sequence."
                    );
            }
        }

        throw std::runtime_error(
            "Metadata JSON contains an unterminated "
            "string."
        );
    }

    std::string parse_number() {
        const std::size_t start =
            position_;

        if (
            !at_end() &&
            peek() == '-'
        ) {
            ++position_;
        }

        if (at_end()) {
            throw std::runtime_error(
                "Metadata JSON contains an incomplete "
                "number."
            );
        }

        if (peek() == '0') {
            ++position_;

            if (
                !at_end() &&
                peek() >= '0' &&
                peek() <= '9'
            ) {
                throw std::runtime_error(
                    "Metadata JSON contains an invalid "
                    "leading zero."
                );
            }
        } else {
            if (
                peek() < '1' ||
                peek() > '9'
            ) {
                throw std::runtime_error(
                    "Metadata JSON contains an invalid "
                    "number."
                );
            }

            while (
                !at_end() &&
                peek() >= '0' &&
                peek() <= '9'
            ) {
                ++position_;
            }
        }

        if (
            !at_end() &&
            peek() == '.'
        ) {
            ++position_;

            if (
                at_end() ||
                peek() < '0' ||
                peek() > '9'
            ) {
                throw std::runtime_error(
                    "Metadata JSON contains an invalid "
                    "fraction."
                );
            }

            while (
                !at_end() &&
                peek() >= '0' &&
                peek() <= '9'
            ) {
                ++position_;
            }
        }

        if (
            !at_end() &&
            (
                peek() == 'e' ||
                peek() == 'E'
            )
        ) {
            ++position_;

            if (
                !at_end() &&
                (
                    peek() == '+' ||
                    peek() == '-'
                )
            ) {
                ++position_;
            }

            if (
                at_end() ||
                peek() < '0' ||
                peek() > '9'
            ) {
                throw std::runtime_error(
                    "Metadata JSON contains an invalid "
                    "exponent."
                );
            }

            while (
                !at_end() &&
                peek() >= '0' &&
                peek() <= '9'
            ) {
                ++position_;
            }
        }

        return std::string(
            text_.substr(
                start,
                position_ -
                    start
            )
        );
    }

    JsonValue parse_array() {
        require_character(
            '['
        );

        JsonValue value;

        value.type =
            JsonValue::Type::Array;

        skip_whitespace();

        if (
            !at_end() &&
            peek() == ']'
        ) {
            ++position_;

            return value;
        }

        while (true) {
            skip_whitespace();

            value.array.push_back(
                parse_value()
            );

            skip_whitespace();

            if (at_end()) {
                throw std::runtime_error(
                    "Metadata JSON contains an "
                    "unterminated array."
                );
            }

            const char separator =
                take();

            if (separator == ']') {
                return value;
            }

            if (separator != ',') {
                throw std::runtime_error(
                    "Metadata JSON contains an invalid "
                    "array separator."
                );
            }

            skip_whitespace();

            if (
                !at_end() &&
                peek() == ']'
            ) {
                throw std::runtime_error(
                    "Metadata JSON contains a trailing "
                    "array comma."
                );
            }
        }
    }

    JsonValue parse_object() {
        require_character(
            '{'
        );

        JsonValue value;

        value.type =
            JsonValue::Type::Object;

        skip_whitespace();

        if (
            !at_end() &&
            peek() == '}'
        ) {
            ++position_;

            return value;
        }

        while (true) {
            skip_whitespace();

            if (
                at_end() ||
                peek() != '"'
            ) {
                throw std::runtime_error(
                    "Metadata JSON object requires a "
                    "string key."
                );
            }

            const std::string key =
                parse_string();

            skip_whitespace();

            require_character(
                ':'
            );

            skip_whitespace();

            if (
                !value.object.emplace(
                    key,
                    parse_value()
                ).second
            ) {
                throw std::runtime_error(
                    "Metadata JSON contains a duplicate "
                    "object key."
                );
            }

            skip_whitespace();

            if (at_end()) {
                throw std::runtime_error(
                    "Metadata JSON contains an "
                    "unterminated object."
                );
            }

            const char separator =
                take();

            if (separator == '}') {
                return value;
            }

            if (separator != ',') {
                throw std::runtime_error(
                    "Metadata JSON contains an invalid "
                    "object separator."
                );
            }

            skip_whitespace();

            if (
                !at_end() &&
                peek() == '}'
            ) {
                throw std::runtime_error(
                    "Metadata JSON contains a trailing "
                    "object comma."
                );
            }
        }
    }

    JsonValue parse_value() {
        skip_whitespace();

        if (at_end()) {
            throw std::runtime_error(
                "Metadata JSON ended unexpectedly."
            );
        }

        switch (peek()) {
            case '{':
                return parse_object();

            case '[':
                return parse_array();

            case '"': {
                JsonValue value;

                value.type =
                    JsonValue::Type::String;

                value.scalar =
                    parse_string();

                return value;
            }

            case 't': {
                require_literal(
                    "true"
                );

                JsonValue value;

                value.type =
                    JsonValue::Type::Boolean;

                value.boolean =
                    true;

                return value;
            }

            case 'f': {
                require_literal(
                    "false"
                );

                JsonValue value;

                value.type =
                    JsonValue::Type::Boolean;

                value.boolean =
                    false;

                return value;
            }

            case 'n':
                require_literal(
                    "null"
                );

                return JsonValue{};

            default: {
                const char first =
                    peek();

                if (
                    first != '-' &&
                    (
                        first < '0' ||
                        first > '9'
                    )
                ) {
                    throw std::runtime_error(
                        "Metadata JSON contains an "
                        "invalid value."
                    );
                }

                JsonValue value;

                value.type =
                    JsonValue::Type::Number;

                value.scalar =
                    parse_number();

                return value;
            }
        }
    }
};

std::string json_escape(
    std::string_view value
) {
    std::string result;

    result.reserve(
        value.size()
    );

    constexpr char digits[] =
        "0123456789abcdef";

    for (unsigned char character :
         value) {
        switch (character) {
            case '"':
                result +=
                    "\\\"";

                break;

            case '\\':
                result +=
                    "\\\\";

                break;

            case '\b':
                result +=
                    "\\b";

                break;

            case '\f':
                result +=
                    "\\f";

                break;

            case '\n':
                result +=
                    "\\n";

                break;

            case '\r':
                result +=
                    "\\r";

                break;

            case '\t':
                result +=
                    "\\t";

                break;

            default:
                if (character < 0x20U) {
                    result +=
                        "\\u00";

                    result.push_back(
                        digits[
                            (
                                character >>
                                4U
                            ) &
                            0x0fU
                        ]
                    );

                    result.push_back(
                        digits[
                            character &
                            0x0fU
                        ]
                    );
                } else {
                    result.push_back(
                        static_cast<char>(
                            character
                        )
                    );
                }

                break;
        }
    }

    return result;
}

std::string json_string(
    std::string_view value
) {
    std::string result =
        "\"";

    result +=
        json_escape(
            value
        );

    result +=
        "\"";

    return result;
}

std::string json_path(
    const std::filesystem::path& path
) {
    return json_string(
        path.generic_string()
    );
}

std::string json_boolean(
    bool value
) {
    return value
        ? "true"
        : "false";
}

std::string json_integer(
    std::int64_t value
) {
    return csv_integer(
        value
    );
}

std::string json_unsigned_integer(
    std::uint64_t value
) {
    return csv_unsigned_integer(
        value
    );
}

std::string json_real(
    double value
) {
    return csv_real(
        value
    );
}

std::string json_optional_uint64(
    const std::optional<std::uint64_t>& value
) {
    if (!value.has_value()) {
        return "null";
    }

    return json_unsigned_integer(
        *value
    );
}

template <class Value, class Formatter>
std::string json_array(
    const std::vector<Value>& values,
    const Formatter& formatter
) {
    std::string result =
        "[";

    for (std::size_t index = 0U;
         index < values.size();
         ++index) {
        if (index != 0U) {
            result +=
                ", ";
        }

        result +=
            formatter(
                values[index]
            );
    }

    result +=
        "]";

    return result;
}

const char* robustness_perturbation_name(
    RobustnessPerturbation perturbation
) {
    switch (perturbation) {
        case RobustnessPerturbation::EdgeDeletion:
            return "edge_deletion";

        case RobustnessPerturbation::EdgeInsertion:
            return "edge_insertion";

        case RobustnessPerturbation::
                DegreePreservingRewiring:
            return
                "degree_preserving_rewiring";

        case RobustnessPerturbation::FiltrationNoise:
            return "filtration_noise";
    }

    throw std::logic_error(
        "Unknown robustness perturbation."
    );
}

const char* experiment_output_status_name(
    ExperimentOutputStatus status
) {
    switch (status) {
        case ExperimentOutputStatus::InProgress:
            return "in_progress";

        case ExperimentOutputStatus::Complete:
            return "complete";

        case ExperimentOutputStatus::Failed:
            return "failed";
    }

    throw std::logic_error(
        "Unknown experiment output status."
    );
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

bool exists_checked(
    const std::filesystem::path& path,
    std::string_view description
) {
    std::error_code error;

    const bool exists =
        std::filesystem::exists(
            path,
            error
        );

    if (error) {
        throw std::runtime_error(
            "Could not inspect " +
            std::string(
                description
            ) +
            " '" +
            path.string() +
            "': " +
            error.message()
        );
    }

    return exists;
}

void remove_checked(
    const std::filesystem::path& path,
    std::string_view description
) {
    std::error_code error;

    const bool removed =
        std::filesystem::remove(
            path,
            error
        );

    if (error) {
        throw std::runtime_error(
            std::string(
                description
            ) +
            " '" +
            path.string() +
            "': " +
            error.message()
        );
    }

    if (!removed) {
        throw std::runtime_error(
            std::string(
                description
            ) +
            " did not remove '" +
            path.string() +
            "'."
        );
    }
}

void rename_checked(
    const std::filesystem::path& source,
    const std::filesystem::path& target,
    std::string_view description
) {
    std::error_code error;

    std::filesystem::rename(
        source,
        target,
        error
    );

    if (error) {
        throw std::runtime_error(
            std::string(
                description
            ) +
            " from '" +
            source.string() +
            "' to '" +
            target.string() +
            "': " +
            error.message()
        );
    }
}

void remove_file_if_present_noexcept(
    const std::filesystem::path& path
) noexcept {
    std::error_code error;

    std::filesystem::remove(
        path,
        error
    );
}

void recover_atomic_output(
    const std::filesystem::path& target,
    std::string_view description
) {
    const std::filesystem::path temporary =
        auxiliary_path(
            target,
            ".tmp"
        );

    const std::filesystem::path backup =
        auxiliary_path(
            target,
            ".bak"
        );

    const bool target_exists =
        exists_checked(
            target,
            description
        );

    if (
        exists_checked(
            backup,
            "metadata backup"
        )
    ) {
        if (target_exists) {
            remove_checked(
                backup,
                "Could not remove stale metadata backup"
            );
        } else {
            rename_checked(
                backup,
                target,
                "Could not restore metadata backup"
            );
        }
    }

    if (
        exists_checked(
            temporary,
            "temporary metadata output"
        )
    ) {
        remove_checked(
            temporary,
            "Could not remove stale temporary metadata"
        );
    }
}

std::string read_text_file(
    const std::filesystem::path& path
) {
    std::ifstream stream(
        path,
        std::ios::in |
            std::ios::binary
    );

    if (!stream.is_open()) {
        throw std::runtime_error(
            "Could not open metadata file '" +
            path.string() +
            "'."
        );
    }

    std::ostringstream content;

    content
        << stream.rdbuf();

    if (
        stream.bad()
    ) {
        throw std::runtime_error(
            "Could not read metadata file '" +
            path.string() +
            "'."
        );
    }

    return content.str();
}

void write_text_atomically(
    const std::filesystem::path& target,
    const std::string& content
) {
    recover_atomic_output(
        target,
        "metadata target"
    );

    const std::filesystem::path temporary =
        auxiliary_path(
            target,
            ".tmp"
        );

    const std::filesystem::path backup =
        auxiliary_path(
            target,
            ".bak"
        );

    try {
        std::ofstream stream(
            temporary,
            std::ios::out |
                std::ios::trunc |
                std::ios::binary
        );

        if (!stream.is_open()) {
            throw std::runtime_error(
                "Could not open temporary metadata file '" +
                temporary.string() +
                "'."
            );
        }

        if (
            content.size() >
            static_cast<std::size_t>(
                std::numeric_limits<
                    std::streamsize
                >::max()
            )
        ) {
            throw std::overflow_error(
                "Metadata output is too large for "
                "stream output."
            );
        }

        stream.write(
            content.data(),
            static_cast<std::streamsize>(
                content.size()
            )
        );

        stream.flush();

        if (!stream) {
            throw std::runtime_error(
                "Could not write metadata output '" +
                temporary.string() +
                "'."
            );
        }

        stream.close();

        if (stream.fail()) {
            throw std::runtime_error(
                "Could not close metadata output '" +
                temporary.string() +
                "'."
            );
        }
    } catch (...) {
        remove_file_if_present_noexcept(
            temporary
        );

        throw;
    }

    if (
        !exists_checked(
            target,
            "metadata target"
        )
    ) {
        rename_checked(
            temporary,
            target,
            "Could not finalize metadata output"
        );

        return;
    }

    if (
        exists_checked(
            backup,
            "metadata backup"
        )
    ) {
        remove_checked(
            backup,
            "Could not remove stale metadata backup"
        );
    }

    rename_checked(
        target,
        backup,
        "Could not preserve existing metadata"
    );

    try {
        rename_checked(
            temporary,
            target,
            "Could not install new metadata"
        );
    } catch (...) {
        remove_file_if_present_noexcept(
            temporary
        );

        try {
            rename_checked(
                backup,
                target,
                "Could not restore previous metadata"
            );
        } catch (...) {
            throw std::runtime_error(
                "Metadata replacement failed, and the "
                "previous metadata could not be restored. "
                "The backup remains at '" +
                backup.string() +
                "'."
            );
        }

        throw;
    }

    remove_checked(
        backup,
        "Could not remove finalized metadata backup"
    );
}

JsonValue parse_json_file(
    const std::filesystem::path& path
) {
    const std::string content =
        read_text_file(
            path
        );

    JsonParser parser(
        content
    );

    return parser.parse();
}

const JsonValue& require_object_member(
    const JsonValue& object,
    std::string_view key
) {
    if (
        object.type !=
        JsonValue::Type::Object
    ) {
        throw std::runtime_error(
            "Metadata JSON requires an object."
        );
    }

    const auto found =
        object.object.find(
            std::string(
                key
            )
        );

    if (
        found ==
        object.object.end()
    ) {
        throw std::runtime_error(
            "Metadata JSON is missing required member '" +
            std::string(
                key
            ) +
            "'."
        );
    }

    return found->second;
}

void require_json_type(
    const JsonValue& value,
    JsonValue::Type expected,
    std::string_view description
) {
    if (
        value.type !=
        expected
    ) {
        throw std::runtime_error(
            std::string(
                description
            ) +
            " has an invalid JSON type."
        );
    }
}

const JsonValue& require_json_object_value(
    const JsonValue& object,
    std::string_view key
) {
    const JsonValue& value =
        require_object_member(
            object,
            key
        );

    require_json_type(
        value,
        JsonValue::Type::Object,
        key
    );

    return value;
}

const JsonValue& require_json_array_value(
    const JsonValue& object,
    std::string_view key
) {
    const JsonValue& value =
        require_object_member(
            object,
            key
        );

    require_json_type(
        value,
        JsonValue::Type::Array,
        key
    );

    return value;
}

const JsonValue& require_json_number_value(
    const JsonValue& object,
    std::string_view key
) {
    const JsonValue& value =
        require_object_member(
            object,
            key
        );

    require_json_type(
        value,
        JsonValue::Type::Number,
        key
    );

    return value;
}

std::string require_json_string(
    const JsonValue& object,
    std::string_view key
) {
    const JsonValue& value =
        require_object_member(
            object,
            key
        );

    require_json_type(
        value,
        JsonValue::Type::String,
        key
    );

    return value.scalar;
}

bool require_json_boolean(
    const JsonValue& object,
    std::string_view key
) {
    const JsonValue& value =
        require_object_member(
            object,
            key
        );

    require_json_type(
        value,
        JsonValue::Type::Boolean,
        key
    );

    return value.boolean;
}

std::int64_t parse_json_int64(
    const JsonValue& value,
    std::string_view description
) {
    require_json_type(
        value,
        JsonValue::Type::Number,
        description
    );

    std::int64_t parsed{};

    const char* const begin =
        value.scalar.data();

    const char* const end =
        begin +
        value.scalar.size();

    const auto result =
        std::from_chars(
            begin,
            end,
            parsed
        );

    if (
        result.ec != std::errc{} ||
        result.ptr != end
    ) {
        throw std::runtime_error(
            std::string(
                description
            ) +
            " is not a valid signed integer."
        );
    }

    return parsed;
}

std::uint64_t parse_json_uint64(
    const JsonValue& value,
    std::string_view description
) {
    require_json_type(
        value,
        JsonValue::Type::Number,
        description
    );

    std::uint64_t parsed{};

    const char* const begin =
        value.scalar.data();

    const char* const end =
        begin +
        value.scalar.size();

    const auto result =
        std::from_chars(
            begin,
            end,
            parsed
        );

    if (
        result.ec != std::errc{} ||
        result.ptr != end
    ) {
        throw std::runtime_error(
            std::string(
                description
            ) +
            " is not a valid unsigned integer."
        );
    }

    return parsed;
}

double parse_json_double(
    const JsonValue& value,
    std::string_view description
) {
    require_json_type(
        value,
        JsonValue::Type::Number,
        description
    );

    double parsed{};

    const char* const begin =
        value.scalar.data();

    const char* const end =
        begin +
        value.scalar.size();

    const auto result =
        std::from_chars(
            begin,
            end,
            parsed,
            std::chars_format::general
        );

    if (
        result.ec != std::errc{} ||
        result.ptr != end ||
        !std::isfinite(
            parsed
        )
    ) {
        throw std::runtime_error(
            std::string(
                description
            ) +
            " is not a finite floating-point value."
        );
    }

    return parsed;
}

void require_int64_equal(
    const JsonValue& object,
    std::string_view key,
    std::int64_t expected
) {
    if (
        parse_json_int64(
            require_json_number_value(
                object,
                key
            ),
            key
        ) !=
        expected
    ) {
        throw std::runtime_error(
            "Existing metadata field '" +
            std::string(
                key
            ) +
            "' does not match the requested experiment."
        );
    }
}

void require_uint64_equal(
    const JsonValue& object,
    std::string_view key,
    std::uint64_t expected
) {
    if (
        parse_json_uint64(
            require_json_number_value(
                object,
                key
            ),
            key
        ) !=
        expected
    ) {
        throw std::runtime_error(
            "Existing metadata field '" +
            std::string(
                key
            ) +
            "' does not match the requested experiment."
        );
    }
}

void require_real_equal(
    const JsonValue& object,
    std::string_view key,
    double expected
) {
    if (
        parse_json_double(
            require_json_number_value(
                object,
                key
            ),
            key
        ) !=
        expected
    ) {
        throw std::runtime_error(
            "Existing metadata field '" +
            std::string(
                key
            ) +
            "' does not match the requested experiment."
        );
    }
}

void require_string_equal(
    const JsonValue& object,
    std::string_view key,
    std::string_view expected
) {
    if (
        require_json_string(
            object,
            key
        ) !=
        expected
    ) {
        throw std::runtime_error(
            "Existing metadata field '" +
            std::string(
                key
            ) +
            "' does not match the requested experiment."
        );
    }
}

void require_optional_uint64_equal(
    const JsonValue& object,
    std::string_view key,
    const std::optional<std::uint64_t>& expected
) {
    const JsonValue& value =
        require_object_member(
            object,
            key
        );

    if (!expected.has_value()) {
        if (
            value.type !=
            JsonValue::Type::Null
        ) {
            throw std::runtime_error(
                "Existing metadata field '" +
                std::string(
                    key
                ) +
                "' does not match the requested "
                "experiment."
            );
        }

        return;
    }

    if (
        parse_json_uint64(
            value,
            key
        ) !=
        *expected
    ) {
        throw std::runtime_error(
            "Existing metadata field '" +
            std::string(
                key
            ) +
            "' does not match the requested experiment."
        );
    }
}

void require_int_array_equal(
    const JsonValue& object,
    std::string_view key,
    const std::vector<int>& expected
) {
    const JsonValue& array =
        require_json_array_value(
            object,
            key
        );

    if (
        array.array.size() !=
        expected.size()
    ) {
        throw std::runtime_error(
            "Existing metadata array '" +
            std::string(
                key
            ) +
            "' does not match the requested experiment."
        );
    }

    for (std::size_t index = 0U;
         index < expected.size();
         ++index) {
        if (
            parse_json_int64(
                array.array[index],
                key
            ) !=
            static_cast<std::int64_t>(
                expected[index]
            )
        ) {
            throw std::runtime_error(
                "Existing metadata array '" +
                std::string(
                    key
                ) +
                "' does not match the requested "
                "experiment."
            );
        }
    }
}

void require_size_array_equal(
    const JsonValue& object,
    std::string_view key,
    const std::vector<std::size_t>& expected
) {
    const JsonValue& array =
        require_json_array_value(
            object,
            key
        );

    if (
        array.array.size() !=
        expected.size()
    ) {
        throw std::runtime_error(
            "Existing metadata array '" +
            std::string(
                key
            ) +
            "' does not match the requested experiment."
        );
    }

    for (std::size_t index = 0U;
         index < expected.size();
         ++index) {
        const std::uint64_t actual =
            parse_json_uint64(
                array.array[index],
                key
            );

        if constexpr (
            sizeof(std::size_t) <
            sizeof(std::uint64_t)
        ) {
            if (
                actual >
                static_cast<std::uint64_t>(
                    std::numeric_limits<
                        std::size_t
                    >::max()
                )
            ) {
                throw std::runtime_error(
                    "Existing metadata array '" +
                    std::string(
                        key
                    ) +
                    "' does not match the requested "
                    "experiment."
                );
            }
        }

        if (
            static_cast<std::size_t>(
                actual
            ) !=
            expected[index]
        ) {
            throw std::runtime_error(
                "Existing metadata array '" +
                std::string(
                    key
                ) +
                "' does not match the requested "
                "experiment."
            );
        }
    }
}

std::string robustness_conditions_json(
    const std::vector<RobustnessCondition>& conditions
) {
    std::string result =
        "[";

    for (std::size_t index = 0U;
         index < conditions.size();
         ++index) {
        if (index != 0U) {
            result +=
                ", ";
        }

        result +=
            "{\"perturbation\": ";

        result +=
            json_string(
                robustness_perturbation_name(
                    conditions[index].
                        perturbation
                )
            );

        result +=
            ", \"severity\": ";

        result +=
            json_real(
                conditions[index].
                    severity
            );

        result +=
            "}";
    }

    result +=
        "]";

    return result;
}

void require_robustness_conditions_equal(
    const JsonValue& robustness,
    const std::vector<RobustnessCondition>& expected
) {
    const JsonValue& conditions =
        require_json_array_value(
            robustness,
            "conditions"
        );

    if (
        conditions.array.size() !=
        expected.size()
    ) {
        throw std::runtime_error(
            "Existing robustness conditions do not "
            "match the requested experiment."
        );
    }

    for (std::size_t index = 0U;
         index < expected.size();
         ++index) {
        const JsonValue& condition =
            conditions.array[index];

        require_json_type(
            condition,
            JsonValue::Type::Object,
            "Robustness condition"
        );

        require_string_equal(
            condition,
            "perturbation",
            robustness_perturbation_name(
                expected[index].
                    perturbation
            )
        );

        require_real_equal(
            condition,
            "severity",
            expected[index].
                severity
        );
    }
}

bool plan_contains_study(
    const ExecutionPlan& plan,
    ExecutionStudy study
) {
    for (const ExecutionTask& task :
         plan.tasks) {
        if (task.study == study) {
            return true;
        }
    }

    return false;
}

std::vector<ExecutionStudy>
execution_plan_studies(
    const ExecutionPlan& plan
) {
    const ExecutionStudy order[] = {
        ExecutionStudy::Baseline,
        ExecutionStudy::
            HarmonicAggregationAblation,
        ExecutionStudy::Robustness,
        ExecutionStudy::
            BatchSizeSensitivity,
        ExecutionStudy::RffSensitivity,
        ExecutionStudy::Scalability
    };

    std::vector<ExecutionStudy> studies;

    for (ExecutionStudy study :
         order) {
        if (
            plan_contains_study(
                plan,
                study
            )
        ) {
            studies.push_back(
                study
            );
        }
    }

    return studies;
}

std::string studies_json(
    const ExecutionPlan& plan
) {
    return json_array(
        execution_plan_studies(
            plan
        ),
        [](
            ExecutionStudy study
        ) {
            return json_string(
                execution_study_name(
                    study
                )
            );
        }
    );
}

void require_studies_equal(
    const JsonValue& manifest,
    const ExecutionPlan& plan
) {
    const JsonValue& stored =
        require_json_array_value(
            manifest,
            "studies"
        );

    const std::vector<ExecutionStudy> expected =
        execution_plan_studies(
            plan
        );

    if (
        stored.array.size() !=
        expected.size()
    ) {
        throw std::runtime_error(
            "Existing manifest studies do not match "
            "the requested execution plan."
        );
    }

    for (std::size_t index = 0U;
         index < expected.size();
         ++index) {
        require_json_type(
            stored.array[index],
            JsonValue::Type::String,
            "Manifest study"
        );

        if (
            stored.array[index].scalar !=
            execution_study_name(
                expected[index]
            )
        ) {
            throw std::runtime_error(
                "Existing manifest studies do not match "
                "the requested execution plan."
            );
        }
    }
}

std::uint64_t task_count_uint64(
    const ExecutionPlan& plan
) {
    if constexpr (
        sizeof(std::size_t) >
        sizeof(std::uint64_t)
    ) {
        if (
            plan.task_count() >
            static_cast<std::size_t>(
                std::numeric_limits<
                    std::uint64_t
                >::max()
            )
        ) {
            throw std::overflow_error(
                "The execution-plan task count does not "
                "fit in std::uint64_t."
            );
        }
    }

    return static_cast<std::uint64_t>(
        plan.task_count()
    );
}

std::string configuration_json_content(
    const ExperimentConfig& config,
    const CanonicalExperimentSpecification& canonical,
    const BuildMetadata& build
) {
    std::ostringstream stream;

    stream
        << "{\n"
        << "  \"schema_version\": "
        << kMetadataSchemaVersion
        << ",\n"
        << "  \"experiment\": "
        << json_string(
               kExperimentName
           )
        << ",\n"
        << "  \"harmonic_aggregation\": {\n"
        << "    \"aggregation_order\": "
        << json_integer(
               kBaselineAggregationOrder
           )
        << ",\n"
        << "    \"highest_explicit_order\": "
        << json_integer(
               kBaselineHighestExplicitOrder
           )
        << "\n"
        << "  },\n"
        << "  \"benchmark\": {\n"
        << "    \"vertex_count\": "
        << json_integer(
               canonical.
                   benchmark.
                   graph_size.
                   vertex_count
           )
        << ",\n"
        << "    \"edge_count\": "
        << json_integer(
               canonical.
                   benchmark.
                   graph_size.
                   edge_count
           )
        << ",\n"
        << "    \"batches_per_model\": "
        << json_integer(
               canonical.
                   benchmark.
                   batches_per_model
           )
        << ",\n"
        << "    \"graphs_per_batch\": "
        << json_integer(
               canonical.
                   benchmark.
                   graphs_per_batch
           )
        << "\n"
        << "  },\n"
        << "  \"random_features\": {\n"
        << "    \"baseline_character_count\": "
        << json_integer(
               canonical.
                   random_features.
                   baseline_character_count
           )
        << ",\n"
        << "    \"replicate_count\": "
        << json_integer(
               canonical.
                   random_features.
                   replicate_count
           )
        << ",\n"
        << "    \"character_counts\": "
        << json_array(
               canonical.
                   random_features.
                   character_counts,
               [](
                   int value
               ) {
                   return json_integer(
                       value
                   );
               }
           )
        << "\n"
        << "  },\n"
        << "  \"robustness\": {\n"
        << "    \"realization_count\": "
        << json_integer(
               canonical.
                   robustness.
                   realization_count
           )
        << ",\n"
        << "    \"conditions\": "
        << robustness_conditions_json(
               canonical.
                   robustness.
                   conditions
           )
        << "\n"
        << "  },\n"
        << "  \"batch_size_sensitivity\": {\n"
        << "    \"graphs_per_batch\": "
        << json_array(
               canonical.
                   batch_size_sensitivity.
                   graphs_per_batch,
               [](
                   int value
               ) {
                   return json_integer(
                       value
                   );
               }
           )
        << "\n"
        << "  },\n"
        << "  \"scalability\": {\n"
        << "    \"graphs_per_batch\": "
        << json_array(
               canonical.
                   scalability.
                   graphs_per_batch,
               [](
                   int value
               ) {
                   return json_integer(
                       value
                   );
               }
           )
        << ",\n"
        << "    \"input_support_sizes\": "
        << json_array(
               canonical.
                   scalability.
                   input_support_sizes,
               [](
                   std::size_t value
               ) {
                   return csv_size(
                       value
                   );
               }
           )
        << ",\n"
        << "    \"character_counts\": "
        << json_array(
               canonical.
                   scalability.
                   character_counts,
               [](
                   int value
               ) {
                   return json_integer(
                       value
                   );
               }
           )
        << ",\n"
        << "    \"repeat_count\": "
        << json_integer(
               canonical.
                   scalability.
                   repeat_count
           )
        << "\n"
        << "  },\n"
        << "  \"cross_observation_ablation_realizations\": "
        << json_integer(
               canonical.
                   cross_observation_ablation_realizations
           )
        << ",\n"
        << "  \"seed\": {\n"
        << "    \"canonical_root_seed\": "
        << json_unsigned_integer(
               canonical.root_seed
           )
        << ",\n"
        << "    \"root_seed_override\": "
        << json_optional_uint64(
               config.root_seed_override
           )
        << ",\n"
        << "    \"effective_root_seed\": "
        << json_unsigned_integer(
               experiment_root_seed(
                   config
               )
           )
        << "\n"
        << "  },\n"
        << "  \"build\": {\n"
        << "    \"created_utc\": "
        << json_string(
               build.created_utc
           )
        << ",\n"
        << "    \"compiler\": "
        << json_string(
               build.compiler
           )
        << ",\n"
        << "    \"compiler_version\": "
        << json_string(
               build.compiler_version
           )
        << ",\n"
        << "    \"build_type\": "
        << json_string(
               build.build_type
           )
        << ",\n"
        << "    \"operating_system\": "
        << json_string(
               build.operating_system
           )
        << ",\n"
        << "    \"executable\": "
        << json_string(
               build.executable
           )
        << "\n"
        << "  }\n"
        << "}\n";

    return stream.str();
}

std::string manifest_json_content(
    const OutputLayout& layout,
    const ExecutionPlan& plan,
    ExperimentOutputStatus status,
    const BuildMetadata& build,
    const std::optional<
        std::string_view
    >& failure_message
) {
    std::ostringstream stream;

    stream
        << "{\n"
        << "  \"schema_version\": "
        << kMetadataSchemaVersion
        << ",\n"
        << "  \"experiment\": "
        << json_string(
               kExperimentName
           )
        << ",\n"
        << "  \"status\": "
        << json_string(
               experiment_output_status_name(
                   status
               )
           )
        << ",\n"
        << "  \"created_utc\": "
        << json_string(
               build.created_utc
           )
        << ",\n"
        << "  \"output_root\": "
        << json_path(
               layout.root()
           )
        << ",\n"
        << "  \"configuration_file\": "
        << json_string(
               "configuration.json"
           )
        << ",\n"
        << "  \"models_file\": "
        << json_string(
               "models.csv"
           )
        << ",\n"
        << "  \"model_parameters_file\": "
        << json_string(
               "model_parameters.csv"
           )
        << ",\n"
        << "  \"task_count\": "
        << json_unsigned_integer(
               task_count_uint64(
                   plan
               )
           )
        << ",\n"
        << "  \"studies\": "
        << studies_json(
               plan
           )
        << ",\n"
        << "  \"failure_message\": ";

    if (failure_message.has_value()) {
        stream
            << json_string(
                   *failure_message
               );
    } else {
        stream
            << "null";
    }

    stream
        << "\n"
        << "}\n";

    return stream.str();
}

using MetadataRow =
    std::vector<std::string>;

std::vector<MetadataRow>
model_metadata_rows(
    const ExecutionPlan& plan
) {
    std::vector<MetadataRow> rows;

    rows.reserve(
        plan.benchmark.models.size()
    );

    for (const BenchmarkModel& model :
         plan.benchmark.models) {
        rows.push_back(
            {
                csv_integer(
                    model.model_index
                ),
                model.descriptor.stable_id,
                model.descriptor.display_name,
                model.descriptor.variant,
                csv_integer(
                    plan.
                        benchmark.
                        graph_size.
                        vertex_count
                ),
                csv_integer(
                    plan.
                        benchmark.
                        graph_size.
                        edge_count
                ),
                csv_size(
                    parameter_condition_count(
                        plan.benchmark,
                        model.descriptor.family
                    )
                )
            }
        );
    }

    return rows;
}

void append_parameter_row(
    std::vector<MetadataRow>& rows,
    const BenchmarkModel& model,
    std::size_t condition_index,
    std::string_view parameter,
    std::string value
) {
    rows.push_back(
        {
            csv_integer(
                model.model_index
            ),
            model.descriptor.stable_id,
            csv_size(
                condition_index
            ),
            std::string(
                parameter
            ),
            std::move(
                value
            )
        }
    );
}

std::vector<MetadataRow>
model_parameter_rows(
    const BenchmarkSpecification& benchmark
) {
    std::vector<MetadataRow> rows;

    for (const BenchmarkModel& model :
         benchmark.models) {
        switch (model.descriptor.family) {
            case RandomNetworkFamily::ErdosRenyi:
                break;

            case RandomNetworkFamily::WattsStrogatz:
                for (
                    std::size_t condition_index = 0U;
                    condition_index <
                        benchmark.
                            model_parameters.
                            watts_strogatz.size();
                    ++condition_index
                ) {
                    const WattsStrogatzParameters&
                        parameters =
                            benchmark.
                                model_parameters.
                                watts_strogatz[
                                    condition_index
                                ];

                    append_parameter_row(
                        rows,
                        model,
                        condition_index,
                        "degree",
                        csv_integer(
                            parameters.degree
                        )
                    );

                    append_parameter_row(
                        rows,
                        model,
                        condition_index,
                        "rewiring_probability",
                        csv_real(
                            parameters.
                                rewiring_probability
                        )
                    );
                }

                break;

            case RandomNetworkFamily::BarabasiAlbert:
                for (
                    std::size_t condition_index = 0U;
                    condition_index <
                        benchmark.
                            model_parameters.
                            barabasi_albert.size();
                    ++condition_index
                ) {
                    const BarabasiAlbertParameters&
                        parameters =
                            benchmark.
                                model_parameters.
                                barabasi_albert[
                                    condition_index
                                ];

                    append_parameter_row(
                        rows,
                        model,
                        condition_index,
                        "attachment_count",
                        csv_integer(
                            parameters.
                                attachment_count
                        )
                    );

                    append_parameter_row(
                        rows,
                        model,
                        condition_index,
                        "initial_clique_size",
                        csv_integer(
                            parameters.
                                initial_clique_size
                        )
                    );
                }

                break;

            case RandomNetworkFamily::ConfigurationModel:
                for (
                    std::size_t condition_index = 0U;
                    condition_index <
                        benchmark.
                            model_parameters.
                            configuration_model.size();
                    ++condition_index
                ) {
                    const ConfigurationModelParameters&
                        parameters =
                            benchmark.
                                model_parameters.
                                configuration_model[
                                    condition_index
                                ];

                    append_parameter_row(
                        rows,
                        model,
                        condition_index,
                        "degree",
                        csv_integer(
                            parameters.degree
                        )
                    );
                }

                break;

            case RandomNetworkFamily::StochasticBlockModel:
                for (
                    std::size_t condition_index = 0U;
                    condition_index <
                        benchmark.
                            model_parameters.
                            stochastic_block_model.size();
                    ++condition_index
                ) {
                    const StochasticBlockModelParameters&
                        parameters =
                            benchmark.
                                model_parameters.
                                stochastic_block_model[
                                    condition_index
                                ];

                    append_parameter_row(
                        rows,
                        model,
                        condition_index,
                        "block_count",
                        csv_integer(
                            parameters.
                                block_count
                        )
                    );

                    append_parameter_row(
                        rows,
                        model,
                        condition_index,
                        "log_odds_contrast",
                        csv_real(
                            parameters.
                                log_odds_contrast
                        )
                    );
                }

                break;

            case RandomNetworkFamily::ChungLu:
                for (
                    std::size_t condition_index = 0U;
                    condition_index <
                        benchmark.
                            model_parameters.
                            chung_lu.size();
                    ++condition_index
                ) {
                    const ChungLuParameters&
                        parameters =
                            benchmark.
                                model_parameters.
                                chung_lu[
                                    condition_index
                                ];

                    append_parameter_row(
                        rows,
                        model,
                        condition_index,
                        "degree_exponent",
                        csv_real(
                            parameters.
                                degree_exponent
                        )
                    );
                }

                break;

            case RandomNetworkFamily::Kleinberg:
                for (
                    std::size_t condition_index = 0U;
                    condition_index <
                        benchmark.
                            model_parameters.
                            kleinberg.size();
                    ++condition_index
                ) {
                    const KleinbergParameters&
                        parameters =
                            benchmark.
                                model_parameters.
                                kleinberg[
                                    condition_index
                                ];

                    append_parameter_row(
                        rows,
                        model,
                        condition_index,
                        "rows",
                        csv_integer(
                            parameters.rows
                        )
                    );

                    append_parameter_row(
                        rows,
                        model,
                        condition_index,
                        "columns",
                        csv_integer(
                            parameters.columns
                        )
                    );

                    append_parameter_row(
                        rows,
                        model,
                        condition_index,
                        "long_range_edge_count",
                        csv_integer(
                            parameters.
                                long_range_edge_count
                        )
                    );

                    append_parameter_row(
                        rows,
                        model,
                        condition_index,
                        "distance_exponent",
                        csv_real(
                            parameters.
                                distance_exponent
                        )
                    );
                }

                break;

            case RandomNetworkFamily::Girg:
                for (
                    std::size_t condition_index = 0U;
                    condition_index <
                        benchmark.
                            model_parameters.
                            girg.size();
                    ++condition_index
                ) {
                    const GirgParameters&
                        parameters =
                            benchmark.
                                model_parameters.
                                girg[
                                    condition_index
                                ];

                    append_parameter_row(
                        rows,
                        model,
                        condition_index,
                        "degree_exponent",
                        csv_real(
                            parameters.
                                degree_exponent
                        )
                    );

                    append_parameter_row(
                        rows,
                        model,
                        condition_index,
                        "connection_exponent",
                        csv_real(
                            parameters.
                                connection_exponent
                        )
                    );
                }

                break;

            case RandomNetworkFamily::
                    HyperbolicRandomGraph:
                for (
                    std::size_t condition_index = 0U;
                    condition_index <
                        benchmark.
                            model_parameters.
                            hyperbolic_random_graph.size();
                    ++condition_index
                ) {
                    const HyperbolicRandomGraphParameters&
                        parameters =
                            benchmark.
                                model_parameters.
                                hyperbolic_random_graph[
                                    condition_index
                                ];

                    append_parameter_row(
                        rows,
                        model,
                        condition_index,
                        "degree_exponent",
                        csv_real(
                            parameters.
                                degree_exponent
                        )
                    );

                    append_parameter_row(
                        rows,
                        model,
                        condition_index,
                        "temperature",
                        csv_real(
                            parameters.temperature
                        )
                    );

                    append_parameter_row(
                        rows,
                        model,
                        condition_index,
                        "curvature",
                        csv_real(
                            parameters.curvature
                        )
                    );

                    append_parameter_row(
                        rows,
                        model,
                        condition_index,
                        "radius",
                        csv_real(
                            parameters.radius
                        )
                    );
                }

                break;

            case RandomNetworkFamily::
                    ExponentialRandomGraphModel:
                for (
                    std::size_t condition_index = 0U;
                    condition_index <
                        benchmark.
                            model_parameters.
                            exponential_random_graph_model.size();
                    ++condition_index
                ) {
                    const ExponentialRandomGraphModelParameters&
                        parameters =
                            benchmark.
                                model_parameters.
                                exponential_random_graph_model[
                                    condition_index
                                ];

                    append_parameter_row(
                        rows,
                        model,
                        condition_index,
                        "gwesp_coefficient",
                        csv_real(
                            parameters.
                                gwesp_coefficient
                        )
                    );

                    append_parameter_row(
                        rows,
                        model,
                        condition_index,
                        "gwesp_decay",
                        csv_real(
                            parameters.
                                gwesp_decay
                        )
                    );

                    append_parameter_row(
                        rows,
                        model,
                        condition_index,
                        "burn_in_sweeps",
                        csv_integer(
                            parameters.
                                burn_in_sweeps
                        )
                    );

                    append_parameter_row(
                        rows,
                        model,
                        condition_index,
                        "sampling_sweeps",
                        csv_integer(
                            parameters.
                                sampling_sweeps
                        )
                    );
                }

                break;
        }
    }

    return rows;
}

std::string csv_content(
    const MetadataRow& header,
    const std::vector<MetadataRow>& rows
) {
    std::ostringstream stream;

    const auto write_row =
        [&stream](
            const MetadataRow& row
        ) {
            for (std::size_t index = 0U;
                 index < row.size();
                 ++index) {
                if (index != 0U) {
                    stream
                        << ",";
                }

                stream
                    << csv_escape(
                           row[index]
                       );
            }

            stream
                << "\n";
        };

    write_row(
        header
    );

    for (const MetadataRow& row :
         rows) {
        write_row(
            row
        );
    }

    return stream.str();
}

std::string expected_models_csv(
    const ExecutionPlan& plan
) {
    return csv_content(
        {
            "model_index",
            "stable_id",
            "display_name",
            "variant",
            "vertex_count",
            "edge_count",
            "parameter_condition_count"
        },
        model_metadata_rows(
            plan
        )
    );
}

std::string expected_model_parameters_csv(
    const ExecutionPlan& plan
) {
    return csv_content(
        {
            "model_index",
            "model_id",
            "parameter_condition_index",
            "parameter",
            "value"
        },
        model_parameter_rows(
            plan.benchmark
        )
    );
}

void require_file_content_equal(
    const std::filesystem::path& path,
    const std::string& expected,
    std::string_view description
) {
    if (
        !exists_checked(
            path,
            description
        )
    ) {
        throw std::runtime_error(
            std::string(
                description
            ) +
            " does not exist: '" +
            path.string() +
            "'."
        );
    }

    if (
        read_text_file(
            path
        ) !=
        expected
    ) {
        throw std::runtime_error(
            std::string(
                description
            ) +
            " does not match the requested experiment: '" +
            path.string() +
            "'."
        );
    }
}

void validate_existing_configuration(
    const JsonValue& configuration,
    const ExperimentConfig& config,
    const CanonicalExperimentSpecification& canonical,
    const BuildMetadata& current_build
) {
    require_json_type(
        configuration,
        JsonValue::Type::Object,
        "Existing configuration metadata"
    );

    require_int64_equal(
        configuration,
        "schema_version",
        kMetadataSchemaVersion
    );

    require_string_equal(
        configuration,
        "experiment",
        kExperimentName
    );

    const JsonValue& harmonic_aggregation =
        require_json_object_value(
            configuration,
            "harmonic_aggregation"
        );

    require_int64_equal(
        harmonic_aggregation,
        "aggregation_order",
        kBaselineAggregationOrder
    );

    require_int64_equal(
        harmonic_aggregation,
        "highest_explicit_order",
        kBaselineHighestExplicitOrder
    );

    const JsonValue& benchmark =
        require_json_object_value(
            configuration,
            "benchmark"
        );

    require_int64_equal(
        benchmark,
        "vertex_count",
        canonical.
            benchmark.
            graph_size.
            vertex_count
    );

    require_int64_equal(
        benchmark,
        "edge_count",
        canonical.
            benchmark.
            graph_size.
            edge_count
    );

    require_int64_equal(
        benchmark,
        "batches_per_model",
        canonical.
            benchmark.
            batches_per_model
    );

    require_int64_equal(
        benchmark,
        "graphs_per_batch",
        canonical.
            benchmark.
            graphs_per_batch
    );

    const JsonValue& random_features =
        require_json_object_value(
            configuration,
            "random_features"
        );

    require_int64_equal(
        random_features,
        "baseline_character_count",
        canonical.
            random_features.
            baseline_character_count
    );

    require_int64_equal(
        random_features,
        "replicate_count",
        canonical.
            random_features.
            replicate_count
    );

    require_int_array_equal(
        random_features,
        "character_counts",
        canonical.
            random_features.
            character_counts
    );

    const JsonValue& robustness =
        require_json_object_value(
            configuration,
            "robustness"
        );

    require_int64_equal(
        robustness,
        "realization_count",
        canonical.
            robustness.
            realization_count
    );

    require_robustness_conditions_equal(
        robustness,
        canonical.
            robustness.
            conditions
    );

    const JsonValue& batch_size =
        require_json_object_value(
            configuration,
            "batch_size_sensitivity"
        );

    require_int_array_equal(
        batch_size,
        "graphs_per_batch",
        canonical.
            batch_size_sensitivity.
            graphs_per_batch
    );

    const JsonValue& scalability =
        require_json_object_value(
            configuration,
            "scalability"
        );

    require_int_array_equal(
        scalability,
        "graphs_per_batch",
        canonical.
            scalability.
            graphs_per_batch
    );

    require_size_array_equal(
        scalability,
        "input_support_sizes",
        canonical.
            scalability.
            input_support_sizes
    );

    require_int_array_equal(
        scalability,
        "character_counts",
        canonical.
            scalability.
            character_counts
    );

    require_int64_equal(
        scalability,
        "repeat_count",
        canonical.
            scalability.
            repeat_count
    );

    require_int64_equal(
        configuration,
        "cross_observation_ablation_realizations",
        canonical.
            cross_observation_ablation_realizations
    );

    const JsonValue& seed =
        require_json_object_value(
            configuration,
            "seed"
        );

    require_uint64_equal(
        seed,
        "canonical_root_seed",
        canonical.root_seed
    );

    require_optional_uint64_equal(
        seed,
        "root_seed_override",
        config.root_seed_override
    );

    require_uint64_equal(
        seed,
        "effective_root_seed",
        experiment_root_seed(
            config
        )
    );

    const JsonValue& build =
        require_json_object_value(
            configuration,
            "build"
        );

    const std::string previous_compiler =
        require_json_string(
            build,
            "compiler"
        );

    const std::string previous_compiler_version =
        require_json_string(
            build,
            "compiler_version"
        );

    const std::string previous_build_type =
        require_json_string(
            build,
            "build_type"
        );

    (void)require_json_string(
        build,
        "created_utc"
    );

    (void)require_json_string(
        build,
        "operating_system"
    );

    (void)require_json_string(
        build,
        "executable"
    );

    if (
        !config.allow_compiler_mismatch &&
        (
            previous_compiler !=
                current_build.compiler ||
            previous_compiler_version !=
                current_build.compiler_version
        )
    ) {
        throw std::runtime_error(
            "The current compiler does not match the "
            "compiler recorded by the experiment."
        );
    }

    if (
        !config.allow_build_type_mismatch &&
        previous_build_type !=
            current_build.build_type
    ) {
        throw std::runtime_error(
            "The current build type does not match the "
            "build type recorded by the experiment."
        );
    }
}

BuildMetadata original_build_metadata(
    const JsonValue& configuration
) {
    const JsonValue& build =
        require_json_object_value(
            configuration,
            "build"
        );

    return BuildMetadata{
        require_json_string(
            build,
            "created_utc"
        ),
        require_json_string(
            build,
            "compiler"
        ),
        require_json_string(
            build,
            "compiler_version"
        ),
        require_json_string(
            build,
            "build_type"
        ),
        require_json_string(
            build,
            "operating_system"
        ),
        require_json_string(
            build,
            "executable"
        )
    };
}

std::string validate_existing_manifest(
    const JsonValue& manifest,
    const ExecutionPlan& plan
) {
    require_json_type(
        manifest,
        JsonValue::Type::Object,
        "Existing manifest metadata"
    );

    require_int64_equal(
        manifest,
        "schema_version",
        kMetadataSchemaVersion
    );

    require_string_equal(
        manifest,
        "experiment",
        kExperimentName
    );

    (void)require_json_string(
        manifest,
        "output_root"
    );

    require_string_equal(
        manifest,
        "configuration_file",
        "configuration.json"
    );

    require_string_equal(
        manifest,
        "models_file",
        "models.csv"
    );

    require_string_equal(
        manifest,
        "model_parameters_file",
        "model_parameters.csv"
    );

    require_uint64_equal(
        manifest,
        "task_count",
        task_count_uint64(
            plan
        )
    );

    require_studies_equal(
        manifest,
        plan
    );

    const std::string created_utc =
        require_json_string(
            manifest,
            "created_utc"
        );

    const std::string status =
        require_json_string(
            manifest,
            "status"
        );

    if (
        status != "in_progress" &&
        status != "complete" &&
        status != "failed"
    ) {
        throw std::runtime_error(
            "Existing manifest metadata has an unknown "
            "experiment status."
        );
    }

    const JsonValue& failure_message =
        require_object_member(
            manifest,
            "failure_message"
        );

    if (status == "failed") {
        require_json_type(
            failure_message,
            JsonValue::Type::String,
            "Manifest failure_message"
        );
    } else {
        require_json_type(
            failure_message,
            JsonValue::Type::Null,
            "Manifest failure_message"
        );
    }

    return created_utc;
}

BuildMetadata validated_resume_build(
    const OutputLayout& layout,
    const ExperimentConfig& config,
    const CanonicalExperimentSpecification& canonical,
    const BuildMetadata& current_build
) {
    recover_atomic_output(
        layout.configuration_json(),
        "configuration metadata"
    );

    if (
        !exists_checked(
            layout.configuration_json(),
            "configuration metadata"
        )
    ) {
        throw std::runtime_error(
            "Resume metadata requires the existing "
            "configuration file."
        );
    }

    const JsonValue configuration =
        parse_json_file(
            layout.configuration_json()
        );

    validate_existing_configuration(
        configuration,
        config,
        canonical,
        current_build
    );

    return original_build_metadata(
        configuration
    );
}

BuildMetadata validate_resume_metadata(
    const OutputLayout& layout,
    const ExperimentConfig& config,
    const CanonicalExperimentSpecification& canonical,
    const ExecutionPlan& plan,
    const BuildMetadata& current_build
) {
    recover_atomic_output(
        layout.configuration_json(),
        "configuration metadata"
    );

    recover_atomic_output(
        layout.manifest_json(),
        "manifest metadata"
    );

    recover_atomic_output(
        layout.models_csv(),
        "model metadata"
    );

    recover_atomic_output(
        layout.model_parameters_csv(),
        "model-parameter metadata"
    );

    if (
        !exists_checked(
            layout.configuration_json(),
            "configuration metadata"
        )
    ) {
        throw std::runtime_error(
            "Resume metadata requires the existing "
            "configuration file."
        );
    }

    if (
        !exists_checked(
            layout.manifest_json(),
            "manifest metadata"
        )
    ) {
        throw std::runtime_error(
            "Resume metadata requires the existing "
            "manifest file."
        );
    }

    const JsonValue configuration =
        parse_json_file(
            layout.configuration_json()
        );

    validate_existing_configuration(
        configuration,
        config,
        canonical,
        current_build
    );

    const BuildMetadata original_build =
        original_build_metadata(
            configuration
        );

    const JsonValue manifest =
        parse_json_file(
            layout.manifest_json()
        );

    const std::string manifest_created_utc =
        validate_existing_manifest(
            manifest,
            plan
        );

    if (
        manifest_created_utc !=
        original_build.created_utc
    ) {
        throw std::runtime_error(
            "The existing manifest and configuration "
            "metadata disagree on the experiment "
            "creation timestamp."
        );
    }

    require_file_content_equal(
        layout.models_csv(),
        expected_models_csv(
            plan
        ),
        "Existing model metadata"
    );

    require_file_content_equal(
        layout.model_parameters_csv(),
        expected_model_parameters_csv(
            plan
        ),
        "Existing model-parameter metadata"
    );

    return original_build;
}

void write_manifest(
    const OutputLayout& layout,
    const ExecutionPlan& plan,
    ExperimentOutputStatus status,
    const BuildMetadata& build,
    const std::optional<
        std::string_view
    >& failure_message
) {
    write_text_atomically(
        layout.manifest_json(),
        manifest_json_content(
            layout,
            plan,
            status,
            build,
            failure_message
        )
    );
}

void write_configuration_metadata(
    const OutputLayout& layout,
    const ExperimentConfig& config,
    const CanonicalExperimentSpecification& canonical,
    const BuildMetadata& build
) {
    write_text_atomically(
        layout.configuration_json(),
        configuration_json_content(
            config,
            canonical,
            build
        )
    );
}

void write_model_metadata(
    const OutputLayout& layout,
    const ExecutionPlan& plan
) {
    write_text_atomically(
        layout.models_csv(),
        expected_models_csv(
            plan
        )
    );

    write_text_atomically(
        layout.model_parameters_csv(),
        expected_model_parameters_csv(
            plan
        )
    );
}

}  // namespace

void write_initial_metadata(
    const OutputLayout& layout,
    const ExperimentConfig& config,
    const CanonicalExperimentSpecification& canonical,
    const ExecutionPlan& plan,
    const BuildMetadata& build
) {
    layout.create_directories();

    if (config.resume) {
        const BuildMetadata original_build =
            validate_resume_metadata(
                layout,
                config,
                canonical,
                plan,
                build
            );

        write_manifest(
            layout,
            plan,
            ExperimentOutputStatus::InProgress,
            original_build,
            std::nullopt
        );

        return;
    }

    write_configuration_metadata(
        layout,
        config,
        canonical,
        build
    );

    write_model_metadata(
        layout,
        plan
    );

    write_manifest(
        layout,
        plan,
        ExperimentOutputStatus::InProgress,
        build,
        std::nullopt
    );
}

void mark_metadata_complete(
    const OutputLayout& layout,
    const ExperimentConfig& config,
    const CanonicalExperimentSpecification& canonical,
    const ExecutionPlan& plan,
    const BuildMetadata& build
) {
    BuildMetadata manifest_build =
        build;

    if (config.resume) {
        manifest_build =
            validated_resume_build(
                layout,
                config,
                canonical,
                build
            );
    }

    write_manifest(
        layout,
        plan,
        ExperimentOutputStatus::Complete,
        manifest_build,
        std::nullopt
    );
}

void mark_metadata_failed(
    const OutputLayout& layout,
    const ExperimentConfig& config,
    const CanonicalExperimentSpecification& canonical,
    const ExecutionPlan& plan,
    const BuildMetadata& build,
    const std::string& failure_message
) {
    if (failure_message.empty()) {
        throw std::invalid_argument(
            "Failed experiment metadata requires a "
            "failure message."
        );
    }

    BuildMetadata manifest_build =
        build;

    if (config.resume) {
        manifest_build =
            validated_resume_build(
                layout,
                config,
                canonical,
                build
            );
    }

    write_manifest(
        layout,
        plan,
        ExperimentOutputStatus::Failed,
        manifest_build,
        std::string_view(
            failure_message
        )
    );
}

}  // namespace vpd