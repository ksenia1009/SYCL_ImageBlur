#include <CL/sycl.hpp>

#include <array>
#include <iostream>
#include <CL/sycl/intel/fpga_extensions.hpp>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <stb.h>
#include <stb_image.h>
#include <stb_image_write.h>

#include <chrono>

using namespace cl::sycl;

constexpr access::mode sycl_read = access::mode::read;
constexpr access::mode sycl_write = access::mode::write;

class ImageBlur;

const int kernel_radius = 3;
int imgWidth, imgHeight, imgChannels, numOfChannels = 4;

int main() {
	const char* filename = "dog.jpg";
	unsigned char* data = stbi_load(filename, &imgWidth, &imgHeight, &imgChannels, numOfChannels);
	std::cout << "imgWidth: " << imgWidth << "\nimgHeight: " << imgHeight << "\nimgChannels: " << imgChannels << "\nnumOfChannels: " << numOfChannels << std::endl;

	range<2> imgRange{static_cast<size_t>(imgHeight), static_cast<size_t>(imgWidth)};
	image<2> input{data, image_channel_order::rgba, image_channel_type::unorm_int8, imgRange};
	buffer<float4, 2> output{imgRange};

	auto kernel = [&](range<2> range, handler& cgh) {
        auto input_accessor = input.get_access<float4, sycl_read>(cgh);
        auto output_accessor = output.get_access<sycl_write>(cgh, range);

        sampler smpl(coordinate_normalization_mode::unnormalized, addressing_mode::clamp_to_edge, filtering_mode::nearest);

        cgh.parallel_for<ImageBlur>(range, [=](item<2> item) {
            auto id = item.get_id();

            float4 sum{};
            for (int i = -kernel_radius; i < kernel_radius; i++) {
                for (int j = -kernel_radius; j < kernel_radius; j++) {
                    int x = static_cast<int>(id[1]) + i;
                    int y = static_cast<int>(id[0]) + j;
                    sum += input_accessor.read(int2{x, y}, smpl);
                }
            }

			output_accessor[id] = (sum / ((2 * kernel_radius + 1) * (2 * kernel_radius + 1)));
        });
    };

	try {
		queue queue{cpu_selector()};

		auto start = std::chrono::steady_clock::now();
        queue.submit(std::bind(kernel, range<2>(imgHeight, imgWidth), std::placeholders::_1));
		auto end = std::chrono::steady_clock::now();
		queue.wait_and_throw();

		std::cout << "Time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() * 3e-09 << std::endl;
    }
    catch (exception e) {
        std::cerr << e.what();
    }

	unsigned char* out_data = new unsigned char[imgHeight * imgWidth * numOfChannels];

    auto acc = output.get_access<sycl_read>();
    for (size_t i = 0; i < imgHeight; i++) {
        for (size_t j = 0; j < imgWidth; j++) {
            unsigned char r, g, b;
            auto pixel = acc[id<2>{i, j}] * 225.f;

            r = static_cast<unsigned char>(pixel.r());
            g = static_cast<unsigned char>(pixel.g());
            b = static_cast<unsigned char>(pixel.b());

            out_data[i * imgHeight * numOfChannels + numOfChannels * j] = r;
			out_data[i * imgHeight * numOfChannels + numOfChannels * j + 1] = g;
			out_data[i * imgHeight * numOfChannels + numOfChannels * j + 2] = b;
            out_data[i * imgHeight * numOfChannels + numOfChannels * j + 3] = 255;
        }
    }

	stbi_write_jpg("dog_blurred.jpg", imgWidth, imgHeight, numOfChannels, out_data, 100);

	stbi_image_free(data);
	stbi_image_free(out_data);
	return 0;
}
