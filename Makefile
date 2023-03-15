.DEFAULT_GOAL := vcfgz_reader_prod

WfingAll = -pedantic -Wall -Wextra -Wcast-align -Wcast-qual -Wctor-dtor-privacy -Wdisabled-optimization -Wformat=2 -Winit-self -Wlogical-op -Wmissing-include-dirs -Wnoexcept -Wold-style-cast -Woverloaded-virtual -Wredundant-decls -Wshadow -Wsign-conversion -Wsign-promo -Wstrict-null-sentinel -Wstrict-overflow=5 -Wswitch-default -Wundef -Wno-unused

src = ./src/vcfgz_reader.cpp
# src = /home/nikita/work/test_gzip_processing_speed/src/vcfmetadata.cpp
olib_dir = ./lib
o = $(olib_dir)/vcfgz_reader.o
lib = $(olib_dir)/vcfgz_reader.lib

zlib_ver = 1.2.11
zlib_name = zlib-$(zlib_ver)
local_zlib_path = lib/$(zlib_name)

define install_zlib_func
	mkdir -p lib
	cd lib ; \
	wget https://zlib.net/fossils/$(zlib_name).tar.gz -O $(zlib_name).tar.gz ; \
	tar -xvzf $(zlib_name).tar.gz 1>/dev/null && echo ' • untarred zlib source code' || echo ' • failed to untarr zlib source code' ; \
	cd $(zlib_name) ; \
	./configure 1>/dev/null && echo ' • configured zlib' || echo ' • failed to configure zlib' ; \
	make 1>/dev/null && echo ' • built zlib' || echo ' • failed to build zlib' ; \
	make install 1>/dev/null && echo ' • installed zlib in selphi' || echo ' • failed to install zlib in selphi'
endef

vcfgz_reader_prod: $(src)
	$(call install_zlib_func)
	mkdir -p $(olib_dir) 2>/dev/null
	g++ -c -fPIC $(src) -Wl,-rpath,$(local_zlib_path) -o $(o) -lpng -lz
	g++ -shared -Wl,-soname,$(lib) -Wl,-rpath,$(local_zlib_path) -o $(lib) $(o) -lz
	chmod -fc a+x $(lib)

vcfgz_reader_dev: $(src)
	$(call install_zlib_func)
	mkdir -p $(olib_dir) 2>/dev/null
	g++ -g $(WfingAll) -c -fPIC $(src) -Wl,-rpath,$(local_zlib_path) -o $(o) -g -lboost_iostreams -lz
	g++ -g -shared -Wl,-soname,$(lib) -Wl,-rpath,$(local_zlib_path) -o $(lib) $(o) -lz
	chmod -fc a+x $(lib)

vcfgz_reader_test: $(src)
	$(call install_zlib_func)
	mkdir -p $(olib_dir) 2>/dev/null
	g++ -g $(WfingAll) $(src) -Wl,-rpath,$(local_zlib_path) -o $(o) -lboost_iostreams -lz
	chmod -fc a+x $(o)

install_zlib:
	$(call install_zlib_func)


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

