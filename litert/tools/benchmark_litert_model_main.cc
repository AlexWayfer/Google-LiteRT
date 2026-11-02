/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/flags/parse.h"  // from @com_google_absl
#include "absl/flags/reflection.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/strings/strip.h"  // from @com_google_absl
#include "litert/tools/benchmark_litert_model.h"
#include "tflite/c/c_api_types.h"
#include "tflite/tools/logging.h"

namespace litert::benchmark {
namespace {

// Returns the name of a `--name[=value]` or `-name[=value]` argument, or an
// empty string if `arg` is not a flag.
absl::string_view FlagName(absl::string_view arg) {
  if (!absl::ConsumePrefix(&arg, "--") && !absl::ConsumePrefix(&arg, "-")) {
    return "";
  }
  return arg.substr(0, arg.find('='));
}

// Returns the arguments of `argv` that are meant for the TFLite benchmark flag
// parser, i.e. everything that Abseil flag parsing did not consume.
std::vector<char*> GetTfLiteBenchmarkArgs(
    int argc, char** argv, const std::vector<char*>& positional_args,
    const std::vector<absl::UnrecognizedFlag>& unrecognized_flags) {
  absl::flat_hash_set<std::string> unrecognized_names;
  for (const absl::UnrecognizedFlag& flag : unrecognized_flags) {
    if (flag.source == absl::UnrecognizedFlag::kFromArgv) {
      unrecognized_names.insert(flag.flag_name);
    }
  }
  // `positional_args` points into `argv`; its first element is the program
  // name.
  absl::flat_hash_set<const char*> positional;
  for (size_t i = 1; i < positional_args.size(); ++i) {
    positional.insert(positional_args[i]);
  }

  std::vector<char*> tflite_argv = {argv[0]};
  for (int i = 1; i < argc; ++i) {
    const absl::string_view flag_name = FlagName(argv[i]);
    // `--help` is handled by Abseil as a usage flag, but the TFLite parser
    // needs it too to print its own usage.
    const bool is_tflite_flag =
        !flag_name.empty() &&
        (unrecognized_names.contains(flag_name) || flag_name == "help");
    // Leftover positional arguments are forwarded so that the TFLite parser
    // reports them as unconsumed.
    if (is_tflite_flag || positional.contains(argv[i])) {
      tflite_argv.push_back(argv[i]);
    }
  }
  return tflite_argv;
}

}  // namespace

int Main(int argc, char** argv) {
  // Benchmark-specific NPU defaults are set before parsing so that flags given
  // on the command line take precedence over them.
  SetBenchmarkNpuFlagDefaults();

  // Abseil flags (e.g. the NPU vendor flags) are parsed first. Whatever Abseil
  // does not recognize is handed to the TFLite benchmark flag parser, so no
  // list of vendor flag prefixes needs to be maintained here.
  std::vector<char*> positional_args;
  std::vector<absl::UnrecognizedFlag> unrecognized_flags;
  absl::ParseAbseilFlagsOnly(argc, argv, positional_args, unrecognized_flags);
  std::vector<char*> new_argv =
      GetTfLiteBenchmarkArgs(argc, argv, positional_args, unrecognized_flags);
  int new_argc = static_cast<int>(new_argv.size());

  bool has_help = false;
  for (int i = 1; i < argc; ++i) {
    if (absl::string_view(argv[i]) == "--help" ||
        absl::string_view(argv[i]) == "-h") {
      has_help = true;
      break;
    }
  }

  if (has_help) {
    std::cout << "\n  NPU Vendor Flags from LiteRT:\n";
    auto flags = absl::GetAllFlags();
    for (const auto& [name, flag] : flags) {
      if (absl::StrContains(flag->Filename(), "flags/vendors/")) {
        // The current value includes the benchmark-specific defaults.
        std::cout << "    --" << flag->Name() << " (" << flag->Help()
                  << "); default: " << flag->CurrentValue() << ";\n";
      }
    }
    std::cout << std::endl;
  }

  TFLITE_LOG(INFO) << "STARTING!";
  BenchmarkLiteRtModel benchmark;
  if (benchmark.Run(new_argc, new_argv.data()) != kTfLiteOk) {
    // If help was requested, tflite benchmark returns kTfLiteError, which is
    // fine
    if (has_help) {
      return EXIT_SUCCESS;
    }
    TFLITE_LOG(ERROR) << "Benchmarking failed.";
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
}  // namespace litert::benchmark

#ifndef BENCHMARK_MODEL_NO_MAIN
int main(int argc, char** argv) { return litert::benchmark::Main(argc, argv); }
#endif
