#include "renderer.hpp"

#include <exception>
#include <iostream>

int main() {
    try {
        tri::Renderer renderer;
        renderer.run();
    } catch (const std::exception &error) {
        std::cerr << "fatal: " << error.what() << "\n";
        return 1;
    }
    return 0;
}
