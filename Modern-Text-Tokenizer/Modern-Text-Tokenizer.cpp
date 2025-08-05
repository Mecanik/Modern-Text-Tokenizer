#include "Modern-Text-Tokenizer.hpp"
#include <chrono>

using namespace std;
using namespace MecanikDev;

void print_separator(const std::string& title) {
	std::cout << "\n" << std::string(50, '=') << std::endl;
	std::cout << "  " << title << std::endl;
	std::cout << std::string(50, '=') << std::endl;
}

void test_basic_tokenization() {
	print_separator("BASIC TOKENIZATION TEST");

	TextTokenizer tokenizer;

	std::vector<std::string> test_texts = {
		"Hello, world!",
		"This is a test sentence.",
		"Natural language processing with C++",
		"The quick brown fox jumps over the lazy dog."
	};

	for (const auto& text : test_texts) {
		auto tokens = tokenizer.tokenize(text);
		std::cout << "Text: \"" << text << "\"" << std::endl;
		std::cout << "Tokens: ";
		for (size_t i = 0; i < tokens.size(); ++i) {
			std::cout << "'" << tokens[i] << "'";
			if (i < tokens.size() - 1) std::cout << ", ";
		}
		std::cout << " (" << tokens.size() << " tokens)" << std::endl << std::endl;
	}
}

void test_distilbert_vocab_loading() {
	print_separator("DISTILBERT VOCABULARY LOADING");

	TextTokenizer tokenizer;

	std::cout << "Loading DistilBERT vocabulary from 'vocab.txt'..." << std::endl;

	if (tokenizer.load_vocab("vocab.txt")) {
		std::cout << "Successfully loaded vocabulary." << std::endl;
		std::cout << "Vocabulary size: " << tokenizer.vocab_size() << " tokens" << std::endl;

		// Show special token IDs
		std::cout << "\nSpecial Token IDs:" << std::endl;
		std::cout << "  [PAD]: " << tokenizer.get_pad_id() << std::endl;
		std::cout << "  [UNK]: " << tokenizer.get_unk_id() << std::endl;
		std::cout << "  [CLS]: " << tokenizer.get_cls_id() << std::endl;
		std::cout << "  [SEP]: " << tokenizer.get_sep_id() << std::endl;
	}
	else {
		std::cout << "Failed to load vocabulary!" << std::endl;
		std::cout << "Make sure 'vocab.txt' exists in the current directory." << std::endl;
		std::cout << "You can download it from:" << std::endl;
		std::cout << "https://huggingface.co/distilbert/distilbert-base-cased-distilled-squad/raw/main/vocab.txt" << std::endl;
		return;
	}
}

void test_encoding_decoding() {
	print_separator("ENCODING & DECODING TEST");

	TextTokenizer tokenizer;

	if (!tokenizer.load_vocab("vocab.txt")) {
		std::cout << "Cannot test encoding/decoding without vocabulary!" << std::endl;
		return;
	}

	// Configure tokenizer
	tokenizer
		// DistilBERT typically uses lowercase
		.set_lowercase(true)
		.set_split_on_punctuation(true)
		.set_keep_punctuation(true);

	std::vector<std::string> test_texts = {
		"Hello world!",
		"This is a test.",
		"Machine learning is awesome.",
		"Natural language processing with transformers."
	};

	for (const auto& text : test_texts) {
		std::cout << "\nOriginal: \"" << text << "\"" << std::endl;

		// Tokenize to strings
		auto tokens = tokenizer.tokenize(text);
		std::cout << "Tokens: ";
		for (size_t i = 0; i < tokens.size(); ++i) {
			std::cout << "'" << tokens[i] << "'";
			if (i < tokens.size() - 1) std::cout << ", ";
		}
		std::cout << std::endl;

		// Encode to token IDs
		auto token_ids = tokenizer.encode(text);
		std::cout << "Token IDs: ";
		for (size_t i = 0; i < token_ids.size(); ++i) {
			std::cout << token_ids[i];
			if (i < token_ids.size() - 1) std::cout << ", ";
		}
		std::cout << std::endl;

		// Decode back to text
		auto decoded = tokenizer.decode(token_ids);
		std::cout << "Decoded: \"" << decoded << "\"" << std::endl;

		// Check if round-trip is successful
		if (decoded.find(text.substr(0, text.find_first_of(".,!?"))) != std::string::npos) {
			std::cout << "Round-trip successful!" << std::endl;
		}
		else {
			std::cout << "Round-trip differences detected" << std::endl;
		}
	}
}

void test_sequence_encoding() {
	print_separator("SEQUENCE ENCODING FOR ML");

	TextTokenizer tokenizer;

	if (!tokenizer.load_vocab("vocab.txt")) {
		std::cout << "Cannot test sequence encoding without vocabulary!" << std::endl;
		return;
	}

	tokenizer
		.set_lowercase(true)
		.set_split_on_punctuation(true)
		.set_keep_punctuation(true);

	std::vector<std::string> test_sentences = {
		"What is machine learning?",
		"Transformers are powerful neural networks.",
		"BERT revolutionized natural language processing."
	};

	std::cout << "Encoding sequences for ML models (max_length=20):" << std::endl;

	for (const auto& sentence : test_sentences) {
		std::cout << "\nSentence: \"" << sentence << "\"" << std::endl;

		// Encode with special tokens for BERT-style models
		auto sequence_ids = tokenizer.encode_sequence(sentence, 20, true);

		std::cout << "Sequence IDs: [";
		for (size_t i = 0; i < sequence_ids.size(); ++i) {
			std::cout << sequence_ids[i];
			if (i < sequence_ids.size() - 1) std::cout << ", ";
		}
		std::cout << "]" << std::endl;
		std::cout << "Length: " << sequence_ids.size() << " tokens" << std::endl;

		// Show what each ID represents
		std::cout << "Token breakdown: ";
		for (size_t i = 0; i < sequence_ids.size(); ++i) {
			int id = sequence_ids[i];
			if (id == tokenizer.get_cls_id()) {
				std::cout << "[CLS]";
			}
			else if (id == tokenizer.get_sep_id()) {
				std::cout << "[SEP]";
			}
			else if (id == tokenizer.get_pad_id()) {
				std::cout << "[PAD]";
			}
			else if (id == tokenizer.get_unk_id()) {
				std::cout << "[UNK]";
			}
			else {
				std::cout << tokenizer.get_token_by_id(id);
			}
			if (i < sequence_ids.size() - 1) std::cout << " ";
		}
		std::cout << std::endl;
	}
}

void test_performance() {
	print_separator("PERFORMANCE TEST");

	TextTokenizer tokenizer;

	if (!tokenizer.load_vocab("vocab.txt")) {
		std::cout << "Cannot test performance without vocabulary!" << std::endl;
		return;
	}

	tokenizer
		.set_lowercase(true)
		.set_split_on_punctuation(true);

	// Create a large test text
	std::string base_text = "Natural language processing is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language.";
	std::string large_text;

	// Repeat the text 1000 times
	for (int i = 0; i < 1000; ++i) {
		large_text += base_text + " ";
	}

	std::cout << "Performance test with " << large_text.size() << " characters" << std::endl;

	// Test tokenization performance
	auto start_time = std::chrono::high_resolution_clock::now();
	auto tokens = tokenizer.tokenize(large_text);
	auto tokenize_time = std::chrono::high_resolution_clock::now();

	// Test encoding performance
	auto token_ids = tokenizer.encode(large_text);
	auto encode_time = std::chrono::high_resolution_clock::now();

	// Test decoding performance
	auto decoded = tokenizer.decode(token_ids);
	auto decode_time = std::chrono::high_resolution_clock::now();

	// Calculate durations
	auto tokenize_duration = std::chrono::duration_cast<std::chrono::microseconds>(tokenize_time - start_time);
	auto encode_duration = std::chrono::duration_cast<std::chrono::microseconds>(encode_time - tokenize_time);
	auto decode_duration = std::chrono::duration_cast<std::chrono::microseconds>(decode_time - encode_time);

	std::cout << "\nResults:" << std::endl;
	std::cout << "  Tokenization: " << tokenize_duration.count() << " μs (" << tokens.size() << " tokens)" << std::endl;
	std::cout << "  Encoding:     " << encode_duration.count() << " μs" << std::endl;
	std::cout << "  Decoding:     " << decode_duration.count() << " μs" << std::endl;

	// Calculate throughput
	double total_time_ms = (tokenize_duration.count() + encode_duration.count() + decode_duration.count()) / 1000.0;
	double throughput_mb_s = (large_text.size() / 1024.0 / 1024.0) / (total_time_ms / 1000.0);

	std::cout << "  Total time:   " << std::fixed << std::setprecision(2) << total_time_ms << " ms" << std::endl;
	std::cout << "  Throughput:   " << std::fixed << std::setprecision(2) << throughput_mb_s << " MB/s" << std::endl;
}

void test_edge_cases() {
	print_separator("EDGE CASES TEST");

	TextTokenizer tokenizer;

	if (tokenizer.load_vocab("vocab.txt")) {
		tokenizer
			.set_lowercase(true)
			.set_split_on_punctuation(true)
			.set_keep_punctuation(true);
	}

	std::vector<std::string> edge_cases = {
		"",									// Empty string
		"   ",								// Only whitespace
		"Hello",							// Single word
		"!!!",								// Only punctuation
		"Hello123World",					// Mixed alphanumeric
		"café naïve résumé",				// Accented characters
		"你好世界",							// Chinese characters
		"🚀🌟💡",							// Emojis
		"C++ vs Python vs Rust",			// Programming languages
		"user@example.com",					// Email
		"https://www.example.com",			// URL
		"It's a beautiful day, isn't it?"	// Contractions
	};

	for (const auto& text : edge_cases) {
		auto tokens = tokenizer.tokenize(text);
		std::cout << "Input: \"" << text << "\"" << std::endl;
		std::cout << "Tokens (" << tokens.size() << "): ";

		for (size_t i = 0; i < tokens.size(); ++i) {
			std::cout << "'" << tokens[i] << "'";
			if (i < tokens.size() - 1) std::cout << ", ";
		}
		std::cout << std::endl;

		if (tokenizer.has_vocab()) {
			auto token_ids = tokenizer.encode(text);
			std::cout << "IDs: ";
			for (size_t i = 0; i < token_ids.size(); ++i) {
				std::cout << token_ids[i];
				if (i < token_ids.size() - 1) std::cout << ", ";
			}
			std::cout << std::endl;
		}
		std::cout << std::endl;
	}
}

int main()
{
	std::cout << "Vocabulary Tokenizer Demo" << std::endl;
	std::cout << "=======================================" << std::endl;

	test_basic_tokenization();
	test_distilbert_vocab_loading();
	test_encoding_decoding();
	test_sequence_encoding();
	test_performance();
	test_edge_cases();

	std::cout << "Demo completed!" << std::endl;
	std::cout << "=======================================" << std::endl;

	return 0;
}
