#ifndef TOMOS_BENCH_LATEX_REPORTER_HPP
#define TOMOS_BENCH_LATEX_REPORTER_HPP

#include <benchmark/benchmark.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <system_error>
#include <vector>

// Shared by every benchmark executable in this tree (in place of
// benchmark::benchmark_main): runs benchmarks exactly as the library's
// default main would (same console output), while additionally writing:
//  - a standalone, compilable LaTeX table of the results, and
//  - a CSV of the same data (one column per counter, easy for tools like
//    MATLAB to load), and
//  - a copy of plot_benchmarks.m, a generic MATLAB script that reads that
//    CSV and renders a bar chart per numeric column,
// all to <directory containing this executable>/output/.

namespace tomos {
namespace bench {

inline std::string latex_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        switch (c) {
            case '_':
            case '%':
            case '&':
            case '#':
            case '$':
            case '{':
            case '}':
                out += '\\';
                out += c;
                break;
            case '\\':
                out += "\\textbackslash{}";
                break;
            default:
                out += c;
        }
    }
    return out;
}

// Quotes a CSV field only if it contains a character that would otherwise
// break column alignment (RFC 4180 style: wrap in quotes, double up any
// embedded quotes).
inline std::string csv_escape(const std::string& s) {
    if (s.find_first_of(",\"\n") == std::string::npos) return s;
    std::string out = "\"";
    for (char c : s) {
        if (c == '"') out += '"';
        out += c;
    }
    out += '"';
    return out;
}

// Delegates console formatting to the library's own ConsoleReporter (so
// interactive output is unchanged), while accumulating every run and
// rewriting BOTH the .tex and .csv from scratch after every completed
// benchmark (ReportRuns fires once per name+args instance, not just once at
// the very end). This costs a trivial full-file rewrite per benchmark point
// (these files are at most a few hundred KB), but means a process killed
// mid-run — e.g. an interactive HPC job hitting its time limit — still
// leaves a valid, complete .tex/.csv reflecting everything that finished
// before the kill, rather than losing all of it (the previous design only
// wrote once, in Finalize(), so a kill anywhere before the very last
// benchmark point lost everything). Used as the sole "display reporter"
// (see RunBenchmarksWithLatexOutput) so the LaTeX/CSV files are always
// written, independent of the --benchmark_out flag (which the library's
// built-in dual-reporter mechanism otherwise requires).
class LatexAndConsoleReporter : public benchmark::BenchmarkReporter {
public:
    LatexAndConsoleReporter(std::filesystem::path tex_path,
                            std::filesystem::path csv_path)
        : tex_path_(std::move(tex_path)), csv_path_(std::move(csv_path)) {}

    bool ReportContext(const Context& context) override {
        return console_.ReportContext(context);
    }

    void ReportRuns(const std::vector<Run>& reports) override {
        console_.ReportRuns(reports);
        for (const auto& run : reports) {
            for (const auto& kv : run.counters) counter_names_.insert(kv.first);
            all_runs_.push_back(run);
        }
        Flush();
    }

    void Finalize() override {
        console_.Finalize();
        Flush();
    }

private:
    void Flush() {
        std::ofstream tex_out(tex_path_, std::ios::trunc);
        std::ofstream csv_out(csv_path_, std::ios::trunc);
        if (!tex_out || !csv_out) return;  // best-effort; don't abort the run over it

        tex_out << "\\documentclass{article}\n"
                   "\\usepackage{booktabs}\n"
                   "\\usepackage[margin=1in]{geometry}\n"
                   "\\begin{document}\n\n"
                   "\\begin{table}[h]\n"
                   "\\centering\n"
                   "\\small\n"
                   "\\begin{tabular}{lrrrl}\n"
                   "\\toprule\n"
                   "Benchmark & Iterations & Time & CPU & Counters \\\\\n"
                   "\\midrule\n";
        for (const auto& run : all_runs_) {
            std::ostringstream counters;
            bool first = true;
            for (const auto& kv : run.counters) {
                if (!first) counters << ", ";
                first = false;
                counters << latex_escape(kv.first) << "=" << kv.second.value;
            }
            const char* unit = benchmark::GetTimeUnitString(run.time_unit);
            tex_out << latex_escape(run.benchmark_name()) << " & "
                    << run.iterations << " & " << run.GetAdjustedRealTime()
                    << "~" << unit << " & " << run.GetAdjustedCPUTime() << "~"
                    << unit << " & " << counters.str() << " \\\\\n";
        }
        tex_out << "\\bottomrule\n"
                   "\\end{tabular}\n"
                   "\\end{table}\n\n"
                   "\\end{document}\n";

        std::vector<std::string> counter_cols(counter_names_.begin(),
                                               counter_names_.end());
        std::sort(counter_cols.begin(), counter_cols.end());

        csv_out << "benchmark,iterations,real_time,cpu_time,time_unit";
        for (const auto& name : counter_cols) csv_out << "," << csv_escape(name);
        csv_out << "\n";

        for (const auto& run : all_runs_) {
            csv_out << csv_escape(run.benchmark_name()) << "," << run.iterations
                    << "," << run.GetAdjustedRealTime() << ","
                    << run.GetAdjustedCPUTime() << ","
                    << csv_escape(benchmark::GetTimeUnitString(run.time_unit));
            for (const auto& name : counter_cols) {
                csv_out << ",";
                auto it = run.counters.find(name);
                if (it != run.counters.end()) csv_out << it->second.value;
            }
            csv_out << "\n";
        }
    }

    std::filesystem::path tex_path_;
    std::filesystem::path csv_path_;
    benchmark::ConsoleReporter console_;
    std::vector<Run> all_runs_;
    std::set<std::string> counter_names_;
};

// Generic MATLAB script copied into every benchmark's output/ directory
// alongside its .tex/.csv: for each numeric, non-identifying column in the
// CSV sitting next to it, renders a bar chart (one bar per benchmark run)
// and saves it as a PNG in that same directory. Kept as a single shared
// script (rather than one tailored per benchmark) so it stays in sync with
// the CSV schema above with no per-executable maintenance.
inline const char* kPlotBenchmarksMatlabScript = R"MATLAB(
function plot_benchmarks()
% plot_benchmarks  Bar-chart every performance column of the CSV(s) sitting
%   next to this script (written by bench_latex_reporter.hpp) into PNGs in
%   the same directory.

    here = fileparts(mfilename('fullpath'));
    csv_files = dir(fullfile(here, '*.csv'));
    if isempty(csv_files)
        error('plot_benchmarks:noCsv', 'No CSV file found next to %s', ...
            mfilename('fullpath'));
    end

    id_cols = {'benchmark', 'iterations', 'time_unit'};
    for k = 1:numel(csv_files)
        csv_path = fullfile(csv_files(k).folder, csv_files(k).name);
        [~, stem, ~] = fileparts(csv_files(k).name);
        plot_one(csv_path, csv_files(k).folder, stem, id_cols);
    end
end

function plot_one(csv_path, out_dir, stem, id_cols)
    T = readtable(csv_path, 'TextType', 'string');
    names = cellstr(T.benchmark);
    x = 1:height(T);

    time_unit = '';
    if any(strcmp('time_unit', T.Properties.VariableNames)) && height(T) > 0
        time_unit = char(T.time_unit(1));
    end

    cols = T.Properties.VariableNames;
    for c = 1:numel(cols)
        col = cols{c};
        if any(strcmp(col, id_cols)) || ~isnumeric(T.(col))
            continue
        end

        fig = figure('Visible', 'off', 'Position', [100, 100, 1400, 600]);
        bar(x, T.(col));
        set(gca, 'XTick', x, 'XTickLabel', names, 'XTickLabelRotation', 60, ...
            'TickLabelInterpreter', 'none');
        if strcmp(col, 'real_time') || strcmp(col, 'cpu_time')
            ylabel(sprintf('%s (%s)', strrep(col, '_', ' '), time_unit));
        else
            ylabel(strrep(col, '_', ' '), 'Interpreter', 'none');
        end
        title(sprintf('%s: %s', stem, col), 'Interpreter', 'none');
        grid on;

        out_path = fullfile(out_dir, sprintf('%s_%s.png', stem, col));
        saveas(fig, out_path);
        close(fig);
    end
end
)MATLAB";

// Resolves the running executable's own directory (via /proc/self/exe,
// falling back to argv[0] if that's unavailable) so results land next to the
// binary regardless of the caller's current working directory, creates an
// output/ subdirectory there, and runs every benchmark registered via
// BENCHMARK(...)/BENCHMARK_TEMPLATE(...), writing results to the console,
// output/<executable-name>.tex, output/<executable-name>.csv, and a copy of
// plot_benchmarks.m (which reads that CSV to render bar charts).
inline int RunBenchmarksWithLatexOutput(int argc, char** argv) {
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;

    std::error_code ec;
    std::filesystem::path exe_path =
        std::filesystem::read_symlink("/proc/self/exe", ec);
    if (ec) exe_path = std::filesystem::absolute(argv[0]);

    std::filesystem::path out_dir = exe_path.parent_path() / "output";
    std::filesystem::create_directories(out_dir);
    std::string stem = exe_path.stem().string();
    std::filesystem::path tex_path = out_dir / (stem + ".tex");
    std::filesystem::path csv_path = out_dir / (stem + ".csv");
    std::filesystem::path m_path = out_dir / "plot_benchmarks.m";

    std::ofstream m_file(m_path);
    if (m_file) {
        m_file << kPlotBenchmarksMatlabScript;
    } else {
        std::cerr << "Warning: could not open " << m_path << " for writing.\n";
    }

    // Verify the output paths are writable up front (same warn-and-fall-back
    // behavior as before) — the reporter itself opens and closes these files
    // repeatedly (once per completed benchmark, not just once at the end),
    // so results survive the process being killed mid-run.
    bool writable = std::ofstream(tex_path).good() && std::ofstream(csv_path).good();
    if (!writable) {
        std::cerr << "Warning: could not open " << tex_path << " or "
                  << csv_path
                  << " for writing; benchmark results will only be printed "
                    "to console.\n";
        benchmark::RunSpecifiedBenchmarks();
    } else {
        LatexAndConsoleReporter reporter(tex_path, csv_path);
        benchmark::RunSpecifiedBenchmarks(&reporter);
        std::cout << "\nLaTeX results written to " << tex_path << "\n";
        std::cout << "CSV results written to " << csv_path << "\n";
        std::cout << "MATLAB plot script written to " << m_path << "\n";
    }
    benchmark::Shutdown();
    return 0;
}

}  // namespace bench
}  // namespace tomos

#define TOMOS_BENCHMARK_MAIN()                              \
    int main(int argc, char** argv) {                       \
        return ::tomos::bench::RunBenchmarksWithLatexOutput(argc, argv); \
    }

#endif  // TOMOS_BENCH_LATEX_REPORTER_HPP
