# Modern C++ Text Tokenizer for NLP and Machine Learning

A high-performance, header-only C++17/20 text tokenizer for NLP and machine learning. Supports UTF-8, vocabulary encoding, and special tokens like [CLS], [SEP]. Ideal for BERT, DistilBERT, and transformer models. No dependencies!

Unlike HuggingFace Tokenizers (Python) or ICU, this is a lightweight C++ alternative with no dependencies.

![CI Status](https://github.com/your-username/your-repo/workflows/CI/badge.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![C++ Standard](https://img.shields.io/badge/C%2B%2B-17%2F20-blue.svg)

## Features

- **Fast**: Zero-copy processing with `std::string_view`
- **UTF-8 Ready**: Proper handling of Unicode without heavy dependencies
- **Configurable**: Fluent API for customizing tokenization behavior
- **Header-Only**: Single file, easy to integrate
- **ASCII Optimized**: Smart handling of ASCII vs UTF-8 characters
- **Modern C++**: Uses C++17/20 features for clean, efficient code
- **Vocabulary Support**: Load/save vocabularies, encode/decode to token IDs
- **Special Tokens**: Support for [CLS], [SEP], [PAD], [UNK] tokens
- **ML Ready**: Sequence encoding for transformer models

## Requirements

- C++17/20 compatible compiler (GCC 7+, Clang 5+, MSVC 2017+)
- No external dependencies

## Quick Start

```cpp
#include "Modern-Text-Tokenizer.hpp"
using namespace MecanikDev;

// Simple tokenization
auto tokens = TextTokenizer::simple_split("Hello, world!");

// Advanced configuration with vocabulary
TextTokenizer tokenizer;

// Load vocabulary file
tokenizer.load_vocab("vocab.txt");

auto token_ids = tokenizer.encode("Hello, world!");

std::string decoded = tokenizer.decode(token_ids);
```

## API Reference

### Basic Usage

```cpp
// Static method for simple whitespace splitting
std::vector<std::string> tokens = TextTokenizer::simple_split(text);

// Full configurability
TextTokenizer tokenizer;
std::vector<std::string> tokens = tokenizer.tokenize(text);
```

### Configuration Methods

All configuration methods return `TextTokenizer&` for method chaining:

```cpp
using namespace MecanikDev;

TextTokenizer tokenizer;
tokenizer
    .set_lowercase(true)           // Convert to lowercase
    .set_keep_punctuation(true)    // Keep punctuation as separate tokens
    .set_split_on_punctuation(true) // Split on punctuation marks
    .add_delimiter(',')            // Add custom delimiter
    .add_delimiters(".,!?")        // Add multiple delimiters
    .set_special_tokens("[UNK]", "[PAD]", "[CLS]", "[SEP]"); // Configure special tokens
```

### Vocabulary Methods

```cpp
// Load vocabulary from file
tokenizer.load_vocab("vocab.txt");

// Build vocabulary from training texts
std::vector<std::string> training_texts = {"Hello world", "Machine learning", ...};
tokenizer.build_vocab_from_text(training_texts, 2, 30000); // min_freq=2, max_size=30000

// Save vocabulary
tokenizer.save_vocab("my_vocab.txt");

// Encoding and decoding
auto token_ids = tokenizer.encode("Hello world");
std::string text = tokenizer.decode(token_ids);

// Sequence encoding for ML models
auto sequence_ids = tokenizer.encode_sequence("Hello world", 512, true); // max_len=512, add_special_tokens=true
```

### Utility Methods

```cpp
// Count tokens without storing them (memory efficient)
size_t count = tokenizer.count_tokens(text);

// Vocabulary information
size_t vocab_size = tokenizer.vocab_size();
bool has_vocab = tokenizer.has_vocab();

// Special token IDs
int unk_id = tokenizer.get_unk_id();
int pad_id = tokenizer.get_pad_id();
int cls_id = tokenizer.get_cls_id();
int sep_id = tokenizer.get_sep_id();
```

## Examples

### Basic Text Processing

```cpp
using namespace MecanikDev;

std::string text = "Natural language processing is amazing!";

// ["Natural", "language", "processing", "is", "amazing!"]
auto tokens = TextTokenizer::simple_split(text);
```

### Building and Using Vocabulary

```cpp
// Create tokenizer and build vocabulary from training data
TextTokenizer tokenizer;
std::vector<std::string> training_texts = {
    "The quick brown fox jumps",
    "Machine learning is fascinating",
    "Natural language processing rocks"
};

tokenizer
    .set_lowercase(true)
    .set_split_on_punctuation(true)
    .build_vocab_from_text(training_texts, 1, 1000);

// Save vocabulary for later use
tokenizer.save_vocab("my_vocab.txt");

// Encode text to token IDs
auto ids = tokenizer.encode("Machine learning rocks!");
// Example: [1, 156, 234, 445, 2] where 1=[CLS], 2=[SEP], etc.

// Decode back to text
std::string decoded = tokenizer.decode(ids);
```

### ML Model Integration

```cpp
// Load pre-trained vocabulary
TextTokenizer tokenizer;
tokenizer.load_vocab("bert_vocab.txt");

// Prepare sequence for BERT-style model
auto input_ids = tokenizer.encode_sequence(
    "Hello world! How are you?", 
    128,    // max_length
    true    // add_special_tokens ([CLS] and [SEP])
);

// Result: [101, 7592, 2088, 999, 2129, 2024, 2017, 1029, 102, ...]
//         [CLS] Hello world !   How   are  you  ?   [SEP] ...
```

### Preprocessing for ML

```cpp
using namespace MecanikDev;

TextTokenizer preprocessor;
preprocessor
    .set_lowercase(true)
    .set_split_on_punctuation(true);

// ["hello", "world"]
auto tokens = preprocessor.tokenize("Hello, World!");
```

### Keeping Punctuation for Analysis

```cpp
TextTokenizer analyzer;
analyzer
    .set_keep_punctuation(true)
    .set_split_on_punctuation(true);

// ["What", "?", "!", "Really", "?"]
auto tokens = analyzer.tokenize("What?! Really?");
```

### Custom Delimiters

```cpp
TextTokenizer csv_tokenizer;
csv_tokenizer.add_delimiters(",;|");

// ["name", "age", "city", "country"]
auto fields = csv_tokenizer.tokenize("name,age;city|country");
```

### Unicode Support

```cpp
using namespace MecanikDev;

std::string multilingual = "Hello 世界 🌍 مرحبا";
auto tokens = TextTokenizer::simple_split(multilingual);
// ["Hello", "世界", "🌍", "مرحبا"]

// Lowercase preserves non-ASCII characters
auto lower_tokens = TextTokenizer()
    .set_lowercase(true)
    .tokenize("Hello 世界");
// ["hello", "世界"] - Chinese characters preserved
```

### Loading DistilBERT Vocabulary

```bash
# Download the DistilBERT vocabulary
curl -o vocab.txt https://huggingface.co/distilbert/distilbert-base-uncased/raw/main/vocab.txt

# Or using wget
wget https://huggingface.co/distilbert/distilbert-base-uncased/raw/main/vocab.txt
```

```cpp
using namespace MecanikDev;

// Load DistilBERT vocabulary
TextTokenizer tokenizer;
if (tokenizer.load_vocab("vocab.txt")) {
    std::cout << "Loaded " << tokenizer.vocab_size() << " tokens" << std::endl;
    
    // Configure for DistilBERT-style tokenization
    tokenizer
        .set_lowercase(true)           // DistilBERT uses lowercase
        .set_split_on_punctuation(true)
        .set_keep_punctuation(true);
    
    // Test encoding
    auto token_ids = tokenizer.encode("Hello, world!");
    // Result: [7592, 1010, 2088, 999] (example IDs)
    
    // Encode with special tokens for ML
    auto sequence = tokenizer.encode_sequence("Hello, world!", 512, true);
    // Result: [101, 7592, 1010, 2088, 999, 102] ([CLS] + tokens + [SEP])
    
    // Decode back
    std::string text = tokenizer.decode(token_ids);
    // Result: "hello , world !"
}
```

## Architecture

### Design Principles

1. **Zero Dependencies**: No ICU, Boost, or other heavy libraries
2. **UTF-8 Safe**: Detects UTF-8 boundaries without corrupting multibyte sequences
3. **ASCII Optimized**: Fast path for ASCII operations (case conversion, punctuation)
4. **Memory Efficient**: Minimal allocations during tokenization
5. **Configurable**: Fluent interface for different use cases

### Performance Characteristics

- **Time Complexity**: O(n) where n is input length
- **Space Complexity**: O(t) where t is number of tokens
- **UTF-8 Handling**: O(1) character boundary detection
- **Memory**: Uses `string_view` for zero-copy input processing

## 🔬 Performance

Benchmark results on a typical text corpus:

```
Performance test with 174000 characters

Results:
  Tokenization: 2159 μs (22000 tokens)
  Encoding:     1900 μs
  Decoding:     430 μs
  Total time:   4.49 ms
  Throughput:   36.97 MB/s
```

*Benchmark on AMD Ryzen 9 5900X, compiled with -O3.*

## Building

### Single File Integration

Simply include the header:

```cpp
#include "Modern-Text-Tokenizer.hpp"
```

### CMake Integration

```cmake
# Add to your CMakeLists.txt
add_executable(your_app main.cpp Modern-Text-Tokenizer.hpp)
target_compile_features(your_app PRIVATE cxx_std_17)
```

### Compilation Example

```bash
g++ -std=c++17 -O3 -o tokenizer_demo main.cpp
clang++ -std=c++17 -O3 -o tokenizer_demo main.cpp
```

## Testing

The included demo shows various tokenization scenarios:

```bash
./tokenizer_demo
```

Expected output includes:
- Basic tokenization examples
- Unicode handling demonstration
- Performance benchmarks
- Configuration examples

## Roadmap

### Planned Features

- [ ] **Regex Support**: Pattern-based tokenization
- [ ] **Streaming API**: Process large files without loading into memory
- [ ] **Parallel Processing**: Multi-threaded batch tokenization
- [ ] **Custom Normalizers**: User-defined text preprocessing
- [ ] **Subword Tokenization**: BPE/WordPiece support
- [ ] **Benchmark Suite**: Comprehensive performance testing

### Future Considerations

- **C++20 Features**: Ranges, concepts, and modules
- **SIMD Optimization**: Vectorized string processing
- **Memory Mapping**: For huge file processing
- **Language Detection**: Automatic handling of different scripts

## Contributing

Contributions welcome! Areas of interest:

1. **Performance Optimization**: SIMD, better algorithms
2. **Unicode Enhancement**: Better normalization without ICU
3. **Testing**: More edge cases and benchmarks
4. **Documentation**: Examples and tutorials

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- Inspired by modern tokenization libraries like HuggingFace Tokenizers
- UTF-8 handling techniques from various C++ Unicode resources
- Performance optimizations learned from high-performance text processing

---

**Star this repo if you find it useful!**