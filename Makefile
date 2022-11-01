.DEFAULT_GOAL := vcfgz_reader_prod

src = ./src/vcfgz_reader.cpp
# src = /home/nikita/work/test_gzip_processing_speed/src/vcfmetadata.cpp
olib_dir = ./modules
o = $(olib_dir)/vcfgz_reader.o
lib = $(olib_dir)/vcfgz_reader.lib

vcfgz_reader_prod: $(src)
	mkdir -p $(olib_dir) 2>/dev/null
	g++ -g -c -fPIC $(src) -o $(o)
	g++ -g -shared -Wl,-soname,$(lib) -o $(lib) $(o)
	chmod -fc a+x $(lib)

vcfgz_reader_dev: $(src)
	mkdir -p $(olib_dir) 2>/dev/null
	g++ -g \
	-pedantic -Wall -Wextra -Wcast-align -Wcast-qual -Wctor-dtor-privacy -Wdisabled-optimization -Wformat=2 -Winit-self -Wlogical-op -Wmissing-include-dirs -Wnoexcept -Wold-style-cast -Woverloaded-virtual -Wredundant-decls -Wshadow -Wsign-conversion -Wsign-promo -Wstrict-null-sentinel -Wstrict-overflow=5 -Wswitch-default -Wundef -Wno-unused \
	-c -fPIC $(src) -o $(o) -g -lboost_iostreams -lz
	g++ -g -shared -Wl,-soname,$(lib) -o $(lib) $(o)
	chmod -fc a+x $(lib)

vcfgz_reader_test: $(src)
	mkdir -p $(olib_dir) 2>/dev/null
	g++ -g \
	-pedantic -Wall -Wextra -Wcast-align -Wcast-qual -Wctor-dtor-privacy -Wdisabled-optimization -Wformat=2 -Winit-self -Wlogical-op -Wmissing-include-dirs -Wnoexcept -Wold-style-cast -Woverloaded-virtual -Wredundant-decls -Wshadow -Wsign-conversion -Wsign-promo -Wstrict-null-sentinel -Wstrict-overflow=5 -Wswitch-default -Wundef -Wno-unused \
	$(src) -o $(o) -g -lboost_iostreams -lz
	chmod -fc a+x $(o)



# src = /home/nikita/work/test_gzip_processing_speed/src/vcfmetadata.cpp
src_rstrd = ./src/extract_IBDs_from_ref_panel_restored-sep14.cpp
olib_dir_rstrd = ./modules
o_rstrd = $(olib_dir_rstrd)/extract_IBDs_from_ref_panel_rstrd-sep14.o
lib_rstrd = $(olib_dir_rstrd)/extract_IBDs_from_ref_panel_rstrd-sep14.lib

IBDs_extractor_restored_sep14_prod: $(src_rstrd)
	mkdir -p $(olib_dir_rstrd) 2>/dev/null
	g++ -g -c -fPIC $(src_rstrd) -o $(o_rstrd)
	g++ -g -shared -Wl,-soname,$(lib_rstrd) -o $(lib_rstrd) $(o_rstrd)
	chmod -fc a+x $(lib_rstrd)

