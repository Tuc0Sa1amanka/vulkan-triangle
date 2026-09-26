#include "renderer.hpp"

#include <cstdint>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string_view>

namespace {

void printUsage(const char *program) {
    std::cout << "usage: " << program << " [--frames N]\n\n"
              << "  --frames N  render N frames and exit, 0 means run until the "
                 "window is closed (default: 200)\n";
}

uint32_t parseFrames(const char *value) {
    const long long parsed = std::stoll(value);
    if (parsed < 0) {
        throw std::invalid_argument("frame count must not be negative");
    }
    return static_cast<uint32_t>(parsed);
}

} // namespace

int main(int argc, char **argv) {
    uint32_t maxFrames = 200;

    try {
        for (int i = 1; i < argc; ++i) {
            const std::string_view arg = argv[i];
            if (arg == "--help" || arg == "-h") {
                printUsage(argv[0]);
                return 0;
            }
            if (arg == "--frames" && i + 1 < argc) {
                const char *value = argv[++i];
                try {
                    maxFrames = parseFrames(value);
                } catch (const std::exception &) {
                    std::cerr << "fatal: '" << value
                              << "' is not a valid frame count\n";
                    return 1;
                }
                continue;
            }
            std::cerr << "fatal: unexpected argument '" << arg << "'\n";
            printUsage(argv[0]);
            return 1;
        }

        tri::Renderer renderer(maxFrames);
        renderer.run();
    } catch (const std::exception &error) {
        std::cerr << "fatal: " << error.what() << "\n";
        return 1;
    }
    return 0;
}
