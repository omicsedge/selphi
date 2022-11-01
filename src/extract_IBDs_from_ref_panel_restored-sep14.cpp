#include <fstream>
#include <iostream>
#include <boost/iostreams/filtering_streambuf.hpp>
#include <boost/iostreams/copy.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <zlib.h> // https://refspecs.linuxbase.org/LSB_3.0.0/LSB-Core-generic/LSB-Core-generic/zlib-gzgets-1.html
#include <bits/stdc++.h>

#define HOMOREF "0|0"
#define N_SAMPLES 3202
#define N_HAPLOIDS N_SAMPLES*2
#define N_CHARS_ROW_MAX N_HAPLOIDS*2 + 4000
#define N_FULL_SEQ_VARIANTS 1647102
#define REF_PANEL "/mnt/science-shared/data/users/adriano/GAIN/ref/20.reference_panel.30x.hg38.3202samples.vcf.gz"


/*
Array is going to be indexed up to total_haploids*n_lines.
But index is limited by upper bound of the `int` type,
    since the standard enforces `int` type for array indexing.
*/
// #define N_LINES_LIMIT (sizeof(int) / 2) / N_HAPLOIDS


// const std::size_t n_chars_row_max = N_HAPLOIDS*2 + 4000;

float avg(int* arr, int start_i, int end_i){
    double sum = 0;
    for (int i = start_i; i < end_i; ++i){
        sum += arr[i];
    }
    return static_cast<float>(sum) / (end_i - start_i);
}

// int count_tabs(std::string s){
//     int count = 0;
//     for (int i = 0; i < s.size(); i++)
//         if (s[i] == '\t')
//             count++;

//     return count;
// }

struct gzReadPastEOF : public std::exception {
   const char * what () const throw () {
      return "attempt to read from a gzFile object past the EOF.";
   }
};

// struct ref_panel_file{
//     std::ifstream file;
//     std::unique_ptr<std::istream> instream;
// };

// ref_panel_file open_ref_panel(){
//     std::ifstream file(REF_PANEL, std::ios_base::in | std::ios_base::binary);
//     boost::iostreams::filtering_streambuf<boost::iostreams::input> inbuf;
//     inbuf.push(boost::iostreams::gzip_decompressor());
//     inbuf.push(file);
//     // Convert streambuf to istream
//     std::istream instream(&inbuf);
//     std::unique_ptr<std::istream> uniq_instream = std::make_unique<std::istream>(instream);
//     return {file, uniq_instream};
// }


std::vector< char > readline( gzFile f ) {
    std::vector< char > v( N_CHARS_ROW_MAX );
    unsigned pos = 0;
    for ( ;; ) {
        if ( gzgets( f, &v[ pos ], v.size() - pos ) == 0 ) {
            // end-of-file or error

            if (gzeof(f)){
                throw gzReadPastEOF();
            }

            int err;
            const char *msg = gzerror( f, &err );
            if ( err != Z_OK ) {
                // handle error
            }
            break;
        }
        unsigned read = strlen( &v[ pos ] );
        if ( v[ pos + read - 1 ] == '\n' ) {
            if ( pos + read >= 2 && v[ pos + read - 2 ] == '\r' ) {
                pos = pos + read - 2;
            } else {
                pos = pos + read - 1;
            }
            break;
        }
        if ( read == 0 || pos + read < v.size() - 1 ) {
            pos = read + pos;
            break;
        }
        pos = v.size() - 1;
        v.resize( v.size() * 2 );
    }
    v.resize( pos );
    return v;
}

// std::vector< char > readline( gzFile f,  std::vector< char > &v) {
//     unsigned pos = 0;
//     v.resize( pos );
//     for ( ;; ) {
//         if ( gzgets( f, &v[ pos ], v.size() - pos ) == 0 ) {
//             // end-of-file or error

//             if (gzeof(f)){
//                 throw gzReadPastEOF();
//             }

//             int err;
//             const char *msg = gzerror( f, &err );
//             if ( err != Z_OK ) {
//                 // handle error
//             }
//             break;
//         }
//         unsigned read = strlen( &v[ pos ] );
//         if ( v[ pos + read - 1 ] == '\n' ) {
//             if ( pos + read >= 2 && v[ pos + read - 2 ] == '\r' ) {
//                 pos = pos + read - 2;
//             } else {
//                 pos = pos + read - 1;
//             }
//             break;
//         }
//         if ( read == 0 || pos + read < v.size() - 1 ) {
//             pos = read + pos;
//             break;
//         }
//         pos = v.size() - 1;
//         v.resize( v.size() * 2 );
//     }
//     v.resize( pos );
//     return v;
// }

// #CHROM  POS     ID      REF     ALT     QUAL    FILTER  INFO    FORMAT  H

uint8_t get_haploid_from_line(
    std::vector<char> line,
    int haploid_i
){
    const int metadata_columns = 9;

    int metadata = 0;
    int i = 0;
    while (true){
        if (i == metadata_columns)
            break;
        if (line[metadata] == '\t')
            i++;
        metadata++;
    }

    return static_cast<uint8_t>(
        line[metadata + (haploid_i * 2)] - '0'
    );

    // return static_cast<uint8_t>(
    //     line[metadata + (haploid_i*2)] - '0'
    // );

    // for (i = 0; i < N_HAPLOIDS; i++){

    // }

    // for (char i : vec){

    // }
}


void populate_array_from_line(
    uint8_t* arr,
    std::vector<char> line,
    long long unsigned int total_haploids,
    long long unsigned int line_i
){
    const int metadata_columns = 9;

    char* line_array = &line[0];

    int metadata = 0;
    int i = 0;
    while (true){
        if (i == metadata_columns)
            break;
        if (line_array[metadata] == '\t')
            i++;
        metadata++;
    }

    long long unsigned int z = line_i*total_haploids;

    for (int i = 0; i < total_haploids; i++){
        *(arr + z + i) = static_cast<uint8_t>(line_array[metadata + (i * 2)] - '0');
    }

    /*
    even/odd                 eoeoeoeoeoeoeoeoeoeoeoeoe
    tens                               111111111122222
    units                    0123456789012345678901234
                             |                      ||
                             v                      vv
    mmmmmmmmmmmmmmmmmmmmmmmmm0|1 0|0 0|0 0|0 1|0 0|1$
                             ^                       ^
                             |                       |
                        int metadata            line.size()

    [space denotes a tab, $ denotes a newline]

    this structure should guarantee `(line.size() - metadata) % 2 == 0`
    */
    // std::size_t n = (line.size()+static_cast<std::size_t>(1)-metadata) / static_cast<std::size_t>(2);
    // for (long long unsigned int i = 0; i < n; i++){
    //     *(arr + z + i) = static_cast<uint8_t>(line[metadata + (i * 2)] - '0');
    // }

    // long long unsigned int  n = static_cast<long long unsigned int>(
    //     (line.size()+static_cast<std::size_t>(1)-metadata)
    //             / static_cast<std::size_t>(2)
    // );
    // for (int i = 0; i < std::min(total_haploids, n); i++){
    //     *(arr + z + i) = static_cast<uint8_t>(line[metadata + (i * 2)] - '0');
    // }


    // return static_cast<uint8_t>(
    //     line[metadata + (haploid_i * 2)] - '0'
    // );

    // arr[]

    // return static_cast<uint8_t>(line[metadata + (haploid_i * 2)]) -
    //        static_cast<uint8_t>('0');

    // return static_cast<uint8_t>(
    //     line[metadata + (haploid_i*2)] - '0'
    // );

    // for (i = 0; i < N_HAPLOIDS; i++){

    // }

    // for (char i : vec){

    // }
}


void populate_array_from_line__int(
    uint8_t* arr,
    std::vector<char> line,
    int total_haploids,
    int line_i
){
    const int metadata_columns = 9;

    char* line_array = &line[0];

    int metadata = 0;
    int i = 0;
    while (true){
        if (i == metadata_columns)
            break;
        if (line_array[metadata] == '\t')
            i++;
        metadata++;
    }

    int z = line_i*total_haploids;



    for (int i = 0; i < total_haploids; i++){
        *(arr + z + i) = static_cast<uint8_t>(line_array[metadata + (i * 2)] - '0');
    }

    /*
    even/odd                 eoeoeoeoeoeoeoeoeoeoeoeoe
    tens                               111111111122222
    units                    0123456789012345678901234
                             |                      ||
                             v                      vv
    mmmmmmmmmmmmmmmmmmmmmmmmm0|1 0|0 0|0 0|0 1|0 0|1$
                             ^                       ^
                             |                       |
                        int metadata            line.size()

    [space denotes a tab, $ denotes a newline]

    this structure should guarantee `(line.size() - metadata) % 2 == 0`
    */
    // std::size_t n = (line.size()+static_cast<std::size_t>(1)-metadata) / static_cast<std::size_t>(2);
    // for (long long unsigned int i = 0; i < n; i++){
    //     *(arr + z + i) = static_cast<uint8_t>(line[metadata + (i * 2)] - '0');
    // }

}




void cout_vect(std::vector<char> vec){
    for (char i : vec)
        std::cout << i;
    std::cout << "[size: " << vec.size() << "]";
    std::cout << std::endl;
}





enum file_state {
    closed,
    opened,
};

struct segment {
    std::vector<int> v;
    int v_size;
    int progress;
};

class IBDs_extractor{
    const char* path;
    gzFile ref_panel;
    std::vector<char> v;
    char* v_arr;
    int header_row_i = 0;
    int table_row_i = 0;
    file_state state = closed;
    bool header_read = false;

    const int metadata_columns = 9;

    private:
        void open(const char* path){
            this->path = path;
            this->ref_panel = gzopen(path, "r");
            this->state = opened;
            std::cout << "file path: " << this->path << std::endl;
            std::cout << "file obj:  " << this->ref_panel << std::endl ;
            this->table_row_i = 0;
            this->header_row_i = 0;
        }
        void read_header(){
            while (true) {
                this->v = readline(this->ref_panel);
                if (this->v[0] == '#'){
                    this->header_row_i++;
                } else {
                    this->header_read = true;
                    break;
                }
            }
            std::cout << "IBD_extractor::read_header():    skipped the header" << std::endl;
        }

    public:
        IBDs_extractor(const char* path){
            this->open(path);

            std::vector<char> v(N_CHARS_ROW_MAX);
            this->v = v;

            std::cout << path << std::endl;
        }
        ~IBDs_extractor(){
            if (this->state == opened){
                this->close();
            }
        }

        void close(){
            if (this->state == opened){
                gzclose(this->ref_panel);
                this->state = closed;
            }
        }

        // segment* readlines(int n){
        int readlines(int n){
            std::vector<int> seg(n);

            if (this->state == closed){
                // segment Seg = { seg, 0, this->table_row_i };
                // return &Seg;
                return this->table_row_i;
            }

            int i = 0;

            try {
                if (! this->header_read)
                    this->read_header();

                for (i = 0; i < n; i++){
                    v = readline(this->ref_panel);
                    seg[i] = get_haploid_from_line(v, 6);
                    // if (i == 0){
                    //     cout_vect(v);
                    // }
                    // lines_count[table_row_i] = v.size();
                    this->table_row_i++;
                }
            } catch (gzReadPastEOF& e) {
                this->close();
            }

            // segment Seg = {seg, i, this->table_row_i};
            // return &Seg;
            return this->table_row_i;
        }

        // segment* readlines(int n){
        int readlines2(uint8_t* segm, int n, int haploid){

            if (!(haploid >= 0 && haploid < N_HAPLOIDS))
                throw std::invalid_argument("`haploid` index has to be an non-negative number less than the number of haploids in the reference panel");

            if (this->state == closed){
                // segment Seg = { seg, 0, this->table_row_i };
                // return &Seg;
                return this->table_row_i;
            }

            int i = 0;

            try {
                if (! this->header_read)
                    this->read_header();

                for (i = 0; i < n; i++){
                    segm[i] = get_haploid_from_line(this->v, haploid);
                    this->table_row_i++;
                    this->v = readline(this->ref_panel);
                    // segm[i] = this->table_row_i;
                    // if (i == 0){
                    //     cout_vect(v);
                    // }
                    // lines_count[table_row_i] = v.size();
                }
            } catch (gzReadPastEOF& e) {
                this->close();
            }

            // segment Seg = {seg, i, this->table_row_i};
            // return &Seg;
            return this->table_row_i;
        }

        int readlines3(
            uint8_t* table,
            long long unsigned int n,
            long long unsigned int total_haploids
        ){
            if (total_haploids < 0)
                throw std::invalid_argument("`total_haploids` has to be an non-negative integer");
            if (n < 0)
                throw std::invalid_argument("`n` has to be an non-negative integer");


            if (this->state == closed)
                return this->table_row_i;

            int i = 0;

            try {
                if (! this->header_read)
                    this->read_header();

                // std::cout << "IBD_extractor::readlines3():    going to populate the array at " << table << std::endl;
                for (i = 0; i < n; i++){
                    // if (i == 10 || i == 100 || i == 1000 || i == 10000 || i % 100000 == 0)
                    //     std::cout << "looping at i = " << i << std::endl;
                    // if (i >= N_FULL_SEQ_VARIANTS-1){
                    //     std::cout << "Looping past " << N_FULL_SEQ_VARIANTS-1;
                    //     std::cout << " (at i = " << i << ")" << std::endl;
                    //     std::cout << "Trying to populate array at table_row_i = " << this->table_row_i << std::endl;
                    //     std::cout << "with v = " << &(this->v) << " , of size " << this->v.size() << std::endl;
                    // }
                    // if (i >= 335330){
                    //     std::cout << "looping at i = " << i << std::endl;
                    // }
                    populate_array_from_line(table, this->v, total_haploids, i);
                    this->table_row_i++;
                    this->v = readline(this->ref_panel);
                }
            } catch (gzReadPastEOF& e) {
                this->close();
            }

            return this->table_row_i;
        }


        int readlines3_int(
            uint8_t* table,
            int n,
            int total_haploids
        ){
            if (total_haploids < 0)
                throw std::invalid_argument("`total_haploids` has to be an non-negative integer");
            if (n < 0)
                throw std::invalid_argument("`n` has to be an non-negative integer");


            if (this->state == closed)
                return this->table_row_i;

            int i = 0;

            try {
                if (! this->header_read)
                    this->read_header();

                std::cout << "IBD_extractor::readlines3():    going to populate the array at " << table << std::endl;
                for (i = 0; i < n; i++){
                    // if (i == 10 || i == 100 || i == 1000 || i == 10000 || i % 100000 == 0)
                    //     std::cout << "looping at i = " << i << std::endl;
                    // if (i >= N_FULL_SEQ_VARIANTS-1){
                    //     std::cout << "Looping past " << N_FULL_SEQ_VARIANTS-1;
                    //     std::cout << " (at i = " << i << ")" << std::endl;
                    //     std::cout << "Trying to populate array at table_row_i = " << this->table_row_i << std::endl;
                    //     std::cout << "with v = " << &(this->v) << " , of size " << this->v.size() << std::endl;
                    // }
                    // if (i >= 335330){
                    //     std::cout << "looping at i = " << i << std::endl;
                    // }
                    populate_array_from_line__int(table, this->v, total_haploids, i);
                    this->table_row_i++;
                    this->v = readline(this->ref_panel);
                }
            } catch (gzReadPastEOF& e) {
                this->close();
            }

            return this->table_row_i;
        }



        int readlines3_selfcontained(
            uint8_t* table,
            long long unsigned int n,
            long long unsigned int total_haploids
        ){
            if (total_haploids < 0)
                throw std::invalid_argument("`total_haploids` has to be an non-negative integer");
            if (n < 0)
                throw std::invalid_argument("`n` has to be an non-negative integer");


            if (this->state == closed)
                return this->table_row_i;

            int i = 0;

            try {
                if (! this->header_read)
                    this->read_header();

                std::cout << "IBD_extractor::readlines3():    going to populate the array at " << table << std::endl;
                int metadata;
                int j;
                long long unsigned int z;

                for (i = 0; i < n; i++){
                    this->v_arr = &this->v[0];
                    metadata = 0;
                    j = 0;

                    while (true){
                        if (j == this->metadata_columns)
                            break;
                        if (v_arr[metadata] == '\t')
                            j++;
                        metadata++;
                    }

                    z = i * total_haploids;
                    for (int i = 0; i < total_haploids; i++){
                        *(table + z + j) = static_cast<uint8_t>(v_arr[metadata + (j * 2)] - '0');
                    }
                    // populate_array_from_line(table, this->v, total_haploids, i);

                    this->table_row_i++;
                    this->v = readline(this->ref_panel);
                }
            } catch (gzReadPastEOF& e) {
                this->close();
            }

            return this->table_row_i;
        }




        void hello(int d, const char * p) {
            std::cout << "Hello: " << d << " -> " << p << std::endl;
        }
};

extern "C" {
    IBDs_extractor* IBDs_extractor_new(const char* path){
        std::cout << "input path: " << path << std::endl;
        return new IBDs_extractor(path);
    }

    int readlines(IBDs_extractor *ibds_extractor, int lines_n){
        return ibds_extractor->readlines(lines_n);
    }

    int readlines2(IBDs_extractor *ibds_extractor, uint8_t* segm, int lines_n, int haploid){
        return ibds_extractor->readlines2(segm, lines_n, haploid);
    }

    int readlines3(IBDs_extractor *ibds_extractor, uint8_t* table, long long unsigned int lines_n, long long unsigned int total_haploids){
        return ibds_extractor->readlines3(table, lines_n, total_haploids);
    }

    int readlines3_int(IBDs_extractor *ibds_extractor, uint8_t* table, int lines_n, int total_haploids){
        return ibds_extractor->readlines3_int(table, lines_n, total_haploids);
    }

    int readlines3_selfcontained(IBDs_extractor *ibds_extractor, uint8_t* table, long long unsigned int lines_n, long long unsigned int total_haploids){
        return ibds_extractor->readlines3_selfcontained(table, lines_n, total_haploids);
    }

    void close_(IBDs_extractor *ibds_extractor){
        ibds_extractor->close();
    }
};


int main(){

    IBDs_extractor f = IBDs_extractor(REF_PANEL);

    // long long unsigned int lines_n = 1647105;
    long long unsigned int lines_n = 1647103;
    long long unsigned int total_haploids_n = 6404;

    std::cout << "lines_n = " << lines_n << std::endl;

    // std::cout << "__INT_MAX__ == " << __INT_MAX__ << std::endl;
    // std::cout << "sizeof(uint8_t) == " << sizeof(uint8_t) << std::endl;

    // std::cout << "(__INT_MAX__ / 2) / N_HAPLOIDS";
    // std::cout << " = (" << __INT_MAX__ << "/" << sizeof(int) << ") / " << N_HAPLOIDS;
    // std::cout << " = " << (__INT_MAX__ / sizeof(int)) << " / " << N_HAPLOIDS;
    // std::cout << " = " << (__INT_MAX__ / sizeof(int)) / N_HAPLOIDS << std::endl;

    std::vector<uint8_t> table_v(total_haploids_n*lines_n);
    uint8_t *table = &table_v[0];

    // long long unsigned int z = 300000LLU * 6404LLU;
    // std::cout << "table + z == " << table <<" + "<<z << " = " << table + z << std::endl;
    //                        z = 320000LLU * 6404LLU;
    // std::cout << "table + z == " << table <<" + "<<z << " = " << table + z << std::endl;
    //                        z = 340000LLU * 6404LLU;
    // std::cout << "table + z == " << table <<" + "<<z << " = " << table + z << std::endl;

    int lines_read_n = f.readlines3(table, lines_n, total_haploids_n);


    std::cout << "lines_read_n = " << lines_read_n << std::endl;

    f.close();
}
