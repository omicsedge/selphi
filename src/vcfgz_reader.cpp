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
#define N_CHARS_ROW_MAX N_HAPLOIDS*2 + 4000 // THIS SHOULD BE DYNAMICALLY DECIDED
// #define N_FULL_SEQ_VARIANTS 1647102



struct gzReadPastEOF : public std::exception {
   const char * what () const throw () {
      return "attempt to read from a gzFile object past the EOF.";
   }
};

struct noGenotypesException : public std::exception {
   const char * what () const throw () {
      return "first row of the table data contains no genotypes (e.g. '0/1', '1|1', etc.)";
   }
};



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



int8_t get_haploid_from_line(
    std::vector<char> line,
    int metadata_columns,
    int haploid_i
){
    int metadata = 0;
    int i = 0;
    while (true){
        if (i == metadata_columns)
            break;
        if (line[metadata] == '\t')
            i++;
        metadata++;
    }

    return static_cast<int8_t>(
        line[metadata + (haploid_i * 2)] - '0'
    );
}


void populate_array_from_line(
    int8_t* arr,
    std::vector<char> line,
    long long int total_haploids,
    int metadata_columns,
    long long int line_i
){
    char* line_array = &line[0];

    int metadata = 0;
    {
        int i = 0;
        while (true){
            if (i == metadata_columns)
                break;
            if (line_array[metadata] == '\t')
                i++;
            metadata++;
        }
    }

    long long int z = line_i*total_haploids;

    for (int i = 0; i < total_haploids; i++){
        *(arr + z + i) = static_cast<int8_t>(line_array[metadata + (i * 2)] - '0');
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
    // for (long long int i = 0; i < n; i++){
    //     *(arr + z + i) = static_cast<int8_t>(line[metadata + (i * 2)] - '0');
    // }

    // long long int  n = static_cast<long long int>(
    //     (line.size()+static_cast<std::size_t>(1)-metadata)
    //             / static_cast<std::size_t>(2)
    // );
    // for (int i = 0; i < std::min(total_haploids, n); i++){
    //     *(arr + z + i) = static_cast<int8_t>(line[metadata + (i * 2)] - '0');
    // }


    // return static_cast<int8_t>(
    //     line[metadata + (haploid_i * 2)] - '0'
    // );

}



void cout_vect(std::vector<char> vec){
    for (char i : vec)
        std::cout << i;
    std::cout << "[size: " << vec.size() << "]";
    std::cout << std::endl;
}




bool is_genotype(const std::vector<char> line, int start){
    return (
        isdigit(line[start]) && 
        (line[start+1] == '/' || line[start+1] == '|' || line[start+1] == '\\') &&
        isdigit(line[start+2])
    );
}




enum file_state {
    closed,
    opened,
};


class vcfgz_reader {
    const char* file_path;
    gzFile ref_panel;
    std::vector<char> line_v;
    char* line_arr;
    int header_row_i = 0;
    int table_row_i = 0;
    file_state state = closed;
    bool header_read = false;

    char* table_header;


    private:
        void open(const char* path){
            this->file_path = path;
            this->ref_panel = gzopen(path, "r");
            this->state = opened;
            this->table_row_i = 0;
            this->header_row_i = 0;
            this->header_read = false;
        }
        void read_vcfheader(){
            while (true) {
                this->line_v = readline(this->ref_panel);

                // vcf header are lines at the beginning that start with two hashes ##
                // table header is the line that comes after the vcf header and start with one hash #
                // next comes the table data
                if (this->line_v[0] == '#'){
                    if (this->line_v[1] != '#'){
                        this->table_header = this->line_arr;
                    }
                    this->header_row_i++;
                } else {
                    this->header_read = true;
                    break;
                }
            }


            this->table_header;
            const std::vector<char> first_data_line = this->line_v;
            const char* first_data_line_arr = this->line_arr;

            bool there_are_genotypes = false;
            int number_of_genotypes = 0;
            this->metadata_columns = 1;
            this->total_haploids = 0;

            std::size_t first_data_line_len = first_data_line.size();

            // std::cout << "this->header_row_i = " << this->header_row_i << std::endl;
            // std::cout << "this->table_header = " << this->table_header << std::endl;
            // std::cout << "this->table_header = " << this->table_header << std::endl;

            // std::cout << "FIRST DATA LINE: ";
            // cout_vect(first_data_line);

            int n_cols = 1;

            std::vector<char> slice;
            for (std::size_t i = 0; i < first_data_line_len; i++){
                if (first_data_line[i] == '\t'){
                    n_cols++;
                    slice = std::vector<char>(first_data_line.begin()+i+1, first_data_line.begin()+i+4);
                    if (
                        ( // next 3 characters are the genotype entry (e.g. '0|0')
                            i+3 < first_data_line_len &&
                            is_genotype(first_data_line, i+1)
                        ) && ( // AND these 3 characters are the entire field
                            i+4 == first_data_line_len ||  
                            first_data_line[i+4] == '\n' ||
                            first_data_line[i+4] == '\r' ||
                            first_data_line[i+4] == '\t'   
                        )
                    ){
                        there_are_genotypes = true;
                        number_of_genotypes++;
                    } else {
                        this->metadata_columns++;
                    }
                }
            }

            // std::cout << "this->metadata_columns = " << this->metadata_columns << std::endl;
            // std::cout << "number_of_genotypes = " << number_of_genotypes << std::endl;

            if (number_of_genotypes == 0){
                throw noGenotypesException();
            }

            this->total_haploids = number_of_genotypes*2;

            // std::cout << "total_haploids   = " << this->total_haploids   << std::endl;
            // std::cout << "metadata_columns = " << this->metadata_columns << std::endl;
        }

        void read_header(){
            while (true) {
                this->line_v = readline(this->ref_panel);
                if (this->line_v[0] == '#'){
                    this->header_row_i++;
                } else {
                    this->header_read = true;
                    break;
                }
            }
        }


    public:
        int metadata_columns = 0;
        long long int total_haploids = 0;


        vcfgz_reader(const char* path){
            this->open(path);

            std::vector<char> v(N_CHARS_ROW_MAX);
            this->line_v = v;
            this->read_vcfheader();
        }
        ~vcfgz_reader(){
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


        int readhaploid(int8_t* segm, int n, int haploid){
            if (!(haploid >= 0 && haploid < N_HAPLOIDS))
                throw std::invalid_argument("`haploid` index has to be an non-negative number less than the number of haploids in the reference panel");

            if (this->state == closed){
                return this->table_row_i;
            }

            int i = 0;

            try {
                if (! this->header_read)
                    this->read_header();

                for (i = 0; i < n; i++){
                    segm[i] = get_haploid_from_line(this->line_v, this->metadata_columns, haploid);
                    this->table_row_i++;
                    this->line_v = readline(this->ref_panel);
                }
            } catch (gzReadPastEOF& e) {
                this->close();
            }

            return this->table_row_i;
        }


        int readlines(
            int8_t* table,
            long long int n
            // long long int total_haploids
        ){
            if (n < 0)
                throw std::invalid_argument("`n` has to be an non-negative integer");

            if (this->state == closed)
                return this->table_row_i;

            int i = 0;

            try {
                if (! this->header_read)
                    this->read_vcfheader();

                for (i = 0; i < n; i++){
                    populate_array_from_line(table, this->line_v, this->total_haploids, this->metadata_columns, i);
                    this->table_row_i++;
                    this->line_v = readline(this->ref_panel);
                }
            } catch (gzReadPastEOF& e) {
                this->close();
            }

            return this->table_row_i;
        }

};


extern "C" {
    vcfgz_reader* vcfgz_reader_new(const char* path){
        return new vcfgz_reader(path);
    }

    long long int get_total_haploids(vcfgz_reader *reader){
        return reader->total_haploids;
    }

    int readlines(vcfgz_reader *reader, int8_t* table, long long int lines_n){
        return reader->readlines(table, lines_n);
    }

    void close_(vcfgz_reader *reader){
        reader->close();
    }
}


// unit test
int main(){

    vcfgz_reader f = vcfgz_reader("/home/nikita/work/test_gzip_processing_speed/toyrefpanel.vcf.gz");

    int lines_n = 2;

    std::vector<int8_t> table_v(f.total_haploids*lines_n);
    int8_t* table = &table_v[0];

    int lines_read_n = f.readlines(table, lines_n);

    std::cout << "f.total_haploids = " << f.total_haploids << std::endl;
    std::cout << "lines_n = " << lines_n << std::endl;
    for (int r=0; r<lines_n; r++){
        std::cout << table_v[r*f.total_haploids];
        for (int c=0; c<f.total_haploids; c++){
            std::cout << ' ' << table_v[r*f.total_haploids + c];
        }
        std::cout << std::endl;
    }

    f.close();
}
