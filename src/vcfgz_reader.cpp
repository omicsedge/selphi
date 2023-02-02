#include <fstream>
#include <iostream>
#include <boost/iostreams/filtering_streambuf.hpp>
#include <boost/iostreams/copy.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/range/adaptor/strided.hpp>
#include <boost/range/algorithm/copy.hpp>
#include <boost/assign.hpp>
#include <zlib.h> // https://refspecs.linuxbase.org/LSB_3.0.0/LSB-Core-generic/LSB-Core-generic/zlib-gzgets-1.html
#include <bits/stdc++.h>





// string length allocated for vcf columns fields
constexpr int COL_LEN[] = {
    4,    // #CHROM  #0
    11,   // POS     #1
    350,  // ID      #2
    350,  // REF     #3
    350,  // ALT     #4
    10,   // QUAL    #5
    100,  // FILTER  #6
    1000, // INFO    #7
    50,   // FORMAT  #8
};

// for vcf FORMAT=GT, 1M cache can store a vcf row for up to about 499k samples
const int CACHE_LINE_LEN = 1000000;
char cache_line[CACHE_LINE_LEN];
char table_header_copy[CACHE_LINE_LEN];
int metadata_columns = 0;
const int* metadata_columns_global_p = &metadata_columns;
const int METADATA_COLUMNS = 9;




template <typename T> // T models Any
struct static_cast_func {
    template <typename T1> // T1 models type statically convertible to T
    T operator()(const T1& x) const { return static_cast<T>(x); }
};



template< typename T, int N >
struct EveryNth {
    bool operator()(const T&) { return m_count++ % N == 0; }
    EveryNth() : m_count(0) {}
    private:
    int m_count;
};



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





// void read_col(int col_i, char* arr_c){
//     // arr_c

//     arr_c[line_i * COL_LEN[col_i] + col_char_i] = cache_line[char_i];

//     switch (col_i){
//         case 0: // #CHROM
//             break;
//         case 1: // POS
//             break;
//         case 2: // ID
//             break;
//         case 3: // REF
//             break;
//         case 4: // ALT
//             break;
//         case 5: // QUAL
//             break;
//         case 6: // FILTER
//             break;
//         case 7: // INFO
//             break;
//         case 8: // FORMAT
//             break;
//         case 9: // genotype data (samples columns)
//             break;
//         default:
//             throw std::invalid_argument("`col_i` has to be from 0 to 9 inclusively (0 to 8 for each vcf metadata column, and 9 for sample genotype columns)");
//             break;
//     }
// }

// void read_col(char* col_name, char* arr_c){
//     if (arr_c == nullptr){

//     } else {

//     }
// }



// class fieldReaderOrSkipper {

//     public:
//         fieldReaderOrSkipper(){};
//         virtual ~fieldReaderOrSkipper() = 0; // just a virtual desctructor - makes the class abstract
//         virtual void operator()(char* cache_line, int& line_i, int& char_i, int& col_i, int& col_char_i);

// };

// class fieldReader: public fieldReaderOrSkipper {
//     char* table_array;

//     public:
//         fieldReader(char* table_array){
//             this->table_array = table_array;
//         }

//         void operator()(char* cache_line, int& line_i, int& char_i, int& col_i, int& col_char_i){
//             while (cache_line[char_i] != '\t'){
//                 this->table_array[line_i * COL_LEN[col_i] + col_char_i] = cache_line[char_i];
//                 col_char_i++;
//                 char_i++;
//             }
//             col_i++;
//             char_i++;
//             col_char_i = 0;
//         }
// };

// class fieldSkipper: public fieldReaderOrSkipper {
//     public:
//         fieldSkipper(){};

//         void operator()(char* cache_line, int& line_i, int& char_i, int& col_i, int& col_char_i){
//             while (cache_line[char_i] != '\t'){
//                 char_i++;
//             }
//             col_i++;
//             char_i++;
//         }
// };


// fieldReaderOrSkipper* fieldReaderOrSkipper_(char* table_array){
//     if (table_array == nullptr){
//         return new fieldSkipper();
//     } else {
//         return new fieldReader(table_array);
//     }
// }


// class fieldLooper {
//     public:
//         fieldLooper(){};
//         // using a separate method instead of operator(), 
//         //  because using the operator() is like calling a method with dot (.),
//         //  while we need to call with an arrow (->) to access inherited functions.
//         virtual void call(char const cache_line[CACHE_LINE_LEN], const long long int& line_i, int& char_i, int& col_i, int& col_char_i){
//             std::cout << "o";
//         };
//         void operator()(char const cache_line[CACHE_LINE_LEN], const long long int& line_i, int& char_i, int& col_i, int& col_char_i){
//             this->call(cache_line, line_i, char_i, col_i, col_char_i);
//         };
// };

class fieldLooper {
    public:
        fieldLooper(){};

        // using a separate method instead of operator(), 
        //  because using the operator() is like calling a method with dot (.),
        //  while we need to call with an arrow (->) to access inherited functions from the base class.
        virtual void call(char const cache_line[CACHE_LINE_LEN], const long long int& line_i, int& char_i, int& col_i, int& col_char_i) {
            while (cache_line[char_i] != '\t'){
                char_i++;
            }
            col_i++;
            char_i++;
            // std::cout << "l";
        }
};

class fieldReader: public fieldLooper {
    char* table_array;

    public:
        fieldReader(char* const table_array){
            this->table_array = table_array;
            // fieldLooper();
        }

        void call(char const cache_line[CACHE_LINE_LEN], const long long int& line_i, int& char_i, int& col_i, int& col_char_i) {
            while (cache_line[char_i] != '\t'){
                this->table_array[line_i * COL_LEN[col_i] + col_char_i] = cache_line[char_i];
                col_char_i++;
                char_i++;
            }
            // this->table_array[line_i * COL_LEN[col_i] + col_char_i] = '\0';
            col_i++;
            char_i++;
            col_char_i = 0;
            // std::cout << "r";
        }
};

fieldLooper* fieldReaderOrSkipper(char* const table_array){
    if (table_array == nullptr){
        // std::cout << " is a nullptr" << "\n";
        return new fieldLooper();
    } else {
        // std::cout << " is NOT a nullptr" << "\n";
        return new fieldReader(table_array);
    }
}



// noop
class genotypesLooper {
    public:
        genotypesLooper(){}
        virtual void call(char const cache_line[CACHE_LINE_LEN], const long long int& line_i, int& starting_char_i, const long long int& total_haploids){
            true;
            // std::cout << "l";
        }
};

class genotypesReader: public genotypesLooper {
    int8_t* table_array;

    public:
        genotypesReader(int8_t* const table_array){
            this->table_array = table_array;
        }

        void call(char const cache_line[CACHE_LINE_LEN], const long long int& line_i, int& starting_char_i, const long long int& total_haploids) override {
            const long long int z = line_i * (total_haploids);
            int8_t *arr_shifted = &this->table_array[z]; // skipped rows above

            const char *cache_line_shifted = &cache_line[starting_char_i]; // skipped metadata columns

            for (int i = 0; i < total_haploids; i++){
                arr_shifted[i] = cache_line_shifted[(i * 2)] - '0';
            }

            // std::cout << "r";
        }
};

genotypesLooper* genotypesReaderOrSkipper(int8_t* table_array){
    if (table_array == nullptr){
        return new genotypesLooper();
    } else {
        return new genotypesReader(table_array);
    }
}

// class metadataFieldsLooper {
    
// }111


struct rowParser {
    fieldLooper* metadata_cols_loopers[METADATA_COLUMNS];
    genotypesLooper genotype_cols_looper;
};

class columnsReader {

    fieldLooper* metadata_cols_loopers[METADATA_COLUMNS];
    genotypesLooper* genotype_cols_looper;
    char* row;

    public:
        columnsReader (
            char* const row,
            char* const    CHR_arr = nullptr,
            char* const    POS_arr = nullptr,
            char* const     ID_arr = nullptr,
            char* const    REF_arr = nullptr,
            char* const    ALT_arr = nullptr,
            char* const   QUAL_arr = nullptr,
            char* const FILTER_arr = nullptr,
            char* const   INFO_arr = nullptr,
            char* const FORMAT_arr = nullptr,
            int8_t* const genotypes_arr = nullptr,
            const long long int total_haploids = 0
        ){

            this->row = row;

            // std::cout << "Checking for nullptr inside columnsReader constructor" << "\n";

            // if (CHR_arr != nullptr)
            //     std::cout << "CHR is not a nullptr" << "\n";
            // if (POS_arr != nullptr)
            //     std::cout << "POS is not a nullptr" << "\n";
            // if (ID_arr != nullptr)
            //     std::cout << "ID is not a nullptr" << "\n";
            // if (REF_arr != nullptr)
            //     std::cout << "REF is not a nullptr" << "\n";
            // if (ALT_arr != nullptr)
            //     std::cout << "ALT is not a nullptr" << "\n";
            // if (QUAL_arr != nullptr)
            //     std::cout << "QUAL is not a nullptr" << "\n";
            // if (FILTER_arr != nullptr)
            //     std::cout << "FILTER is not a nullptr" << "\n";
            // if (INFO_arr != nullptr)
            //     std::cout << "INFO is not a nullptr" << "\n";
            // if (FORMAT_arr != nullptr)
            //     std::cout << "FORMAT is not a nullptr" << "\n";
            // if (genotypes_arr != nullptr)
            //     std::cout << "genotypes is not a nullptr" << "\n";

            // int count = 0;
 
            // static const int metadata_columns = 9;
            // // this->metadata_cols_loopers = {
            // //     *fieldReaderOrSkipper(CHR_arr),
            // //     *fieldReaderOrSkipper(POS_arr),
            // //     *fieldReaderOrSkipper(ID_arr),
            // //     *fieldReaderOrSkipper(REF_arr),
            // //     *fieldReaderOrSkipper(ALT_arr),
            // //     *fieldReaderOrSkipper(QUAL_arr),
            // //     *fieldReaderOrSkipper(FILTER_arr),
            // //     *fieldReaderOrSkipper(INFO_arr),
            // //     *fieldReaderOrSkipper(FORMAT_arr)
            // // };
            // std::cout << "CHR";
            this->metadata_cols_loopers[0] = fieldReaderOrSkipper(CHR_arr);
            // std::cout << "POS";
            this->metadata_cols_loopers[1] = fieldReaderOrSkipper(POS_arr);
            // std::cout << "ID";
            this->metadata_cols_loopers[2] = fieldReaderOrSkipper(ID_arr);
            // std::cout << "REF";
            this->metadata_cols_loopers[3] = fieldReaderOrSkipper(REF_arr);
            // std::cout << "ALT";
            this->metadata_cols_loopers[4] = fieldReaderOrSkipper(ALT_arr);
            // std::cout << "QUAL";
            this->metadata_cols_loopers[5] = fieldReaderOrSkipper(QUAL_arr);
            // std::cout << "FILTER";
            this->metadata_cols_loopers[6] = fieldReaderOrSkipper(FILTER_arr);
            // std::cout << "INFO";
            this->metadata_cols_loopers[7] = fieldReaderOrSkipper(INFO_arr);
            // std::cout << "FORMAT";
            this->metadata_cols_loopers[8] = fieldReaderOrSkipper(FORMAT_arr);

            // genotypesLooper* genotype_cols_looper = genotypesReaderOrSkipper(genotypes_arr);
            this->genotype_cols_looper = genotypesReaderOrSkipper(genotypes_arr);

        }


        void operator()(const long long int total_haploids,
                        const long long int line_i)
        {
            int char_i = 0;
            int col_i = 0;
            int col_char_i = 0;

            // read or skip the metadata columns
            for (int i=0; i<METADATA_COLUMNS; i++){
                this->metadata_cols_loopers[i]->call(this->row, line_i, char_i, col_i, col_char_i);
            }
            // std::cout << " ";

            // read or skip the rest: the genotype columns
            this->genotype_cols_looper->call(this->row, line_i, char_i, total_haploids);
            // std::cout << std::endl;
        }

    /*
    THE FOLLOWING HAS TO BE EXECUTED ONCE FOR ALL *N* LINES.

    ideas:
        1. if performance of all null sucks, then add a check:
            "if all pointers to metadata columns are nullptr,
            then just use the codeblock from `populate_array_from_line`"

    */







    // /*
    // THE FOLLOWING HAS TO BE EXECUTED FOR EACH LINE
    // */

    // int char_i = 0;
    // int col_i = 0;
    // int col_char_i = 0;

    // // 1. read or skip metadata columns
    // for (int i=0; i<metadata_columns; i++){
    //     metadata_cols_loopers[i](cache_line, line_i, char_i, col_i, col_char_i);
    // }

    // const int metadata_chars = char_i;


    // // // 2. maybe read genotype columns

    // // const long long int z = line_i * (total_haploids);
    // // int8_t *genotypes_arr_shifted = &genotypes_arr[z]; // skipped rows above

    // // const char* cache_line_shifted = &cache_line[metadata]; // skipped metadata columns

    // // for (int i = 0; i < total_haploids; i++){
    // //     arr_shifted[i] = cache_line_shifted[(i * 2)] - '0';
    // // }
};

// class table_columns_query {

    

//     void operator()(){
        
//     }
// };




int readline_arr( gzFile f, char* cache_line ) {
    // std::vector< char > v( N_CHARS_ROW_MAX );
    unsigned pos = 0;
    unsigned size = 0;
    for ( ;; ) {
        if ( gzgets( f, &cache_line[ pos ], CACHE_LINE_LEN - pos ) == 0 ) {
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
        unsigned read = strlen( &cache_line[ pos ] );
        if ( cache_line[ pos + read - 1 ] == '\n' ) {
            if ( pos + read >= 2 && cache_line[ pos + read - 2 ] == '\r' ) {
                pos = pos + read - 2;
            } else {
                pos = pos + read - 1;
            }
            break;
        }
        if ( read == 0 || pos + read < CACHE_LINE_LEN - 1 ) {
            pos = read + pos;
            break;
        }
        pos = CACHE_LINE_LEN - 1;
        size = CACHE_LINE_LEN * 2;
    }
    size = pos;
    // cache_line[size] = '\0';
    return size;
}





int8_t get_haploid_from_line(
    int haploid_i
){
    const int metadata_cols = *metadata_columns_global_p;

    int metadata = 0;
    int i = 0;
    while (true){
        if (i == metadata_cols)
            break;
        if (cache_line[metadata] == '\t')
            i++;
        metadata++;
    }

    return static_cast<int8_t>(
        cache_line[metadata + (haploid_i * 2)] - '0'
    );
}



void populate_array_from_line(
    int8_t* arr,
    const char* cache_line,
    const long long int total_haploids,
    const long long int line_i
){
    const int metadata_cols = *metadata_columns_global_p;

    int metadata = 0;
    {
        int i = 0;
        while (true){
            if (cache_line[metadata] == '\t')
                i++;
                if (i == metadata_cols)
                    break;
            metadata++;
        }
        metadata++;
    }

    const long long int z = line_i * (total_haploids);
    int8_t* arr_shifted = &arr[z]; // skipped rows above

    const char* cache_line_shifted = &cache_line[metadata]; // skipped metadata columns

    for (int i = 0; i < total_haploids; i++){
        arr_shifted[i] = cache_line_shifted[(i * 2)] - '0';
    }


}



bool is_genotype(const std::vector<char> line, int start){
    return (
        isdigit(line[start]) && 
        (line[start+1] == '/' || line[start+1] == '|' || line[start+1] == '\\') &&
        isdigit(line[start+2])
    );
}






void fill_in_CHR_POS_VID_from_line(
    char* CHR_arr,
    char* POS_arr,
    char* VID_arr,
    const long long int line_i
){

    const int metadata_cols = *metadata_columns_global_p;

    int char_i = 0;
    int col_i = 0;
    int col_char_i = 0;
    
    while (true){
        if (cache_line[char_i] == '\t'){
            col_i++;
            char_i++;
            col_char_i = 0;
            if (col_i == 4)
                break;
            continue;
        } else if (col_i == 0) {
            CHR_arr[line_i*COL_LEN[0] + col_char_i] = cache_line[char_i];
        } else if (col_i == 1) {
            POS_arr[line_i*COL_LEN[1] + col_char_i] = cache_line[char_i];
        } else if (col_i == 2) {
            if (col_char_i < COL_LEN[2]){
                VID_arr[line_i*COL_LEN[2] + col_char_i] = cache_line[char_i];
            }
        } else {
            break;
        }
        col_char_i++;
        char_i++;
    }



    // test this kind of code. This largely may work except e.g. off-by-1 errors
    while (true){
        while (cache_line[char_i] != '\t'){ // looping through chars in a field
            CHR_arr[line_i * COL_LEN[col_i] + col_char_i] = cache_line[char_i];
            col_char_i++;
            char_i++;
        }
        col_i++;
        char_i++;
        col_char_i = 0;
        if (col_i == 9)
            break;
        // continue;
    }

    // samples genotypes columns
    while (true){

    }

    // const long long int z = line_i * (total_haploids);
    // int8_t* arr_shifted = &arr[z]; // skipped rows above

    // const char* cache_line_shifted = &cache_line[metadata]; // skipped metadata columns

    // for (int i = 0; i < total_haploids; i++){
    //     arr_shifted[i] = cache_line_shifted[(i * 2)] - '0';
    // }
}






enum file_state {
    closed,
    opened,
};


struct vcfTableParams {
    const int * metadata_cols;
    const long long * total_haps;
};



class vcfgz_reader {
    const char* file_path;
    gzFile ref_panel;
    char* line_arr;
    int header_row_i = 0;
    int table_row_i = 0;
    file_state state = closed;
    bool header_read = false;

    const int* metadata_columns_p;
    const long long* total_haploids_p;

    int line_len;
    int first_data_line_len;


    private:
        void open(const char* path){
            this->file_path = path;
            this->ref_panel = gzopen(path, "r");
            this->state = opened;
            this->table_row_i = 0;
            this->header_row_i = 0;
            this->header_read = false;
        }

        vcfTableParams read_vcfheader(){

            while (true) {
                this->line_len = readline_arr(this->ref_panel, this->cache_line);

                // vcf header are lines at the beginning that start with two hashes ##
                // table header is the line that comes after the vcf header and start with one hash #
                // next comes the table data
                if (this->cache_line[0] == '#'){
                    if (this->cache_line[1] != '#'){
                        // for (int i = 0; i<this->line_len; i++)
                        //     this->table_header_copy[i] = this->cache_line[i];

                        for (int i = 0; i<this->line_len; i++)
                            this->table_header_copy[i] = this->cache_line[i];

                        this->table_header = this->cache_line;

                        // std::cout << std::endl;
                        // for (int i = 0; i<this->line_len; i++)
                        //     std::cout << this->table_header[i];
                        // std::cout << std::endl;
                        // std::cout << '[' << this->line_len << ']';
                        // std::cout << std::endl;

                        this->table_header_len = this->line_len;
                    }
                    this->header_row_i++;
                } else {
                    this->header_read = true;
                    break;
                }
            }

            const std::vector<char> first_data_line(this->cache_line, this->cache_line+this->line_len);
            this->first_data_line_len = this->line_len;

            bool there_are_genotypes = false;
            int number_of_genotypes = 0;
            this->metadata_columns = 1;
            this->total_haploids = 0;

            int n_cols = 1;

            std::vector<char> slice;
            for (std::size_t i = 0; i < this->first_data_line_len; i++){
                if (first_data_line[i] == '\t'){
                    n_cols++;
                    slice = std::vector<char>(first_data_line.begin()+i+1, first_data_line.begin()+i+4);
                    if (
                        ( // next 3 characters are the genotype entry (e.g. '0|0')
                            i+3 < this->first_data_line_len &&
                            is_genotype(first_data_line, i+1)
                        ) && ( // AND these 3 characters are the entire field
                            i+4 == this->first_data_line_len ||  
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

            if (number_of_genotypes == 0){
                throw noGenotypesException();
            }

            this->total_haploids = number_of_genotypes*2;

            static const long long total_haps = this->total_haploids;

            static const int metadata_cols = this->metadata_columns;
            metadata_columns_global_p = &this->metadata_columns;

            // std::cout << "this->total_haploids = " << this->total_haploids << std::endl;

            return { &metadata_cols, &total_haps };

        }

        // void read_header(){
        //     while (true) {
        //         this->line_v = readline(this->ref_panel);
        //         if (this->line_v[0] == '#'){
        //             this->header_row_i++;
        //         } else {
        //             this->header_read = true;
        //             break;
        //         }
        //     }
        // }


        // int readline_arr( gzFile f ) {
        //     // std::vector< char > v( N_CHARS_ROW_MAX );
        //     unsigned pos = 0;
        //     unsigned size = 0;
        //     for ( ;; ) {
        //         if ( gzgets( f, &this->cache_line[ pos ], CACHE_LINE_LEN - pos ) == 0 ) {
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
        //         unsigned read = strlen( &this->cache_line[ pos ] );
        //         if ( this->cache_line[ pos + read - 1 ] == '\n' ) {
        //             if ( pos + read >= 2 && this->cache_line[ pos + read - 2 ] == '\r' ) {
        //                 pos = pos + read - 2;
        //             } else {
        //                 pos = pos + read - 1;
        //             }
        //             break;
        //         }
        //         if ( read == 0 || pos + read < CACHE_LINE_LEN - 1 ) {
        //             pos = read + pos;
        //             break;
        //         }
        //         pos = CACHE_LINE_LEN - 1;
        //         size = CACHE_LINE_LEN * 2;
        //     }
        //     size = pos;
        //     return size;
        // }



        // void populate_array_from_line(
        //     int8_t* arr,
        //     const long long int total_haploids,
        //     const long long int line_i
        // ){
        //     int metadata = 0;
        //     {
        //         int i = 0;
        //         while (true){
        //             if (this->cache_line[metadata] == '\t')
        //                 i++;
        //                 if (i == METADATA_COLUMNS)
        //                     break;
        //             metadata++;
        //         }
        //         metadata++;
        //     }

        //     const long long int z = line_i * (total_haploids);
        //     int8_t* arr_shifted = &arr[z]; // skipped rows above

        //     const char* cache_line_shifted = &this->cache_line[metadata]; // skipped metadata columns

        //     for (int i = 0; i < total_haploids; i++){
        //         arr_shifted[i] = cache_line_shifted[(i * 2)] - '0';
        //     }

        // }



    public:
        int metadata_columns = 0;
        long long int total_haploids = 0;


        char* table_header;
        // char table_header_copy[CACHE_LINE_LEN];
        int table_header_len;

        const int CACHE_LINE_LEN_prop = CACHE_LINE_LEN;

        char* cache_line;
        char* table_header_copy;



        vcfgz_reader(const char* path){
            this->open(path);

            this->cache_line = new char[CACHE_LINE_LEN];
            this->table_header_copy = new char[CACHE_LINE_LEN];
            // std::unique_ptr<int[]> stuff(new int[CACHE_LINE_LEN]);

            vcfTableParams params = this->read_vcfheader();
            this->metadata_columns_p = params.metadata_cols;
            this->total_haploids_p = params.total_haps;
        }
        ~vcfgz_reader(){
            if (this->state == opened){
                this->close();
            }
            delete [] this->cache_line;
            delete [] this->table_header_copy;
        }

        void close(){
            if (this->state == opened){
                gzclose(this->ref_panel);
                this->state = closed;
            }
        }


        int readhaploid(int8_t* segm, int n, int haploid){
            if (!(haploid >= 0))
                throw std::invalid_argument("`haploid` index has to be an non-negative integer");

            if (this->state == closed){
                return this->table_row_i;
            }

            int i = 0;

            try {
                if (! this->header_read)
                    this->read_vcfheader();

                for (i = 0; i < n; i++){
                    segm[i] = get_haploid_from_line(haploid);
                    this->table_row_i++;
                    this->line_len = readline_arr(this->ref_panel, this->cache_line);
                }
            } catch (gzReadPastEOF& e) {
                this->close();
            }

            return this->table_row_i;
        }


        int readlines(
            int8_t* table,
            long long int n
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
                    populate_array_from_line(table, this->cache_line, this->total_haploids, i);
                    this->table_row_i++;
                    this->line_len = readline_arr(this->ref_panel, this->cache_line);
                }
            } catch (gzReadPastEOF& e) {
                this->close();
            }

            return this->table_row_i;
        }

        int readcolumns(const long long int n,
                        char* const CHR_arr = nullptr,
                        char* const POS_arr = nullptr,
                        char* const ID_arr = nullptr,
                        char* const REF_arr = nullptr,
                        char* const ALT_arr = nullptr,
                        char* const QUAL_arr = nullptr,
                        char* const FILTER_arr = nullptr,
                        char* const INFO_arr = nullptr,
                        char* const FORMAT_arr = nullptr,
                        int8_t* const genotypes_arr = nullptr,
                        const long long int total_haploids = 0)
        {

            // std::cout << "Checking for nullptr inside vcfgz_reader::readcolumns" << "\n";

            // if (CHR_arr != nullptr)
            //     std::cout << "CHR is not a nullptr" << "\n";
            // if (POS_arr != nullptr)
            //     std::cout << "POS is not a nullptr" << "\n";
            // if (ID_arr != nullptr)
            //     std::cout << "ID is not a nullptr" << "\n";
            // if (REF_arr != nullptr)
            //     std::cout << "REF is not a nullptr" << "\n";
            // if (ALT_arr != nullptr)
            //     std::cout << "ALT is not a nullptr" << "\n";
            // if (QUAL_arr != nullptr)
            //     std::cout << "QUAL is not a nullptr" << "\n";
            // if (FILTER_arr != nullptr)
            //     std::cout << "FILTER is not a nullptr" << "\n";
            // if (INFO_arr != nullptr)
            //     std::cout << "INFO is not a nullptr" << "\n";
            // if (FORMAT_arr != nullptr)
            //     std::cout << "FORMAT is not a nullptr" << "\n";
            // if (genotypes_arr != nullptr)
            //     std::cout << "genotypes is not a nullptr" << "\n";

            if (n < 0)
                throw std::invalid_argument("`n` has to be an non-negative integer");

            if (this->state == closed)
                return this->table_row_i;


            columnsReader read_columns = columnsReader(
                this->cache_line,
                CHR_arr,
                POS_arr,
                ID_arr,
                REF_arr,
                ALT_arr,
                QUAL_arr,
                FILTER_arr,
                INFO_arr,
                FORMAT_arr,
                genotypes_arr,
                total_haploids
            );

            try {
                if (! this->header_read)
                    this->read_vcfheader();

                for (int i = 0; i < n; i++){
                    read_columns(total_haploids, i);
                    this->table_row_i++;
                    this->line_len = readline_arr(this->ref_panel, this->cache_line);
                }
            } catch (gzReadPastEOF& e) {
                this->close();
            }

            return this->table_row_i;
        }

        int readsites(
            char* CHR_arr,
            char* POS_arr,
            char* VID_arr,
            long long int n
        ){
            // std::cout << "cpp: hello vcfgz_reader::readsites()" << std::endl;
            if (n < 0)
                throw std::invalid_argument("`n` has to be an non-negative integer");

            if (this->state == closed)
                return this->table_row_i;

            int i = 0;

            // std::cout << "cpp: vcfgz_reader::readsites() : before try statement" << std::endl;
            try {
                if (! this->header_read)
                    this->read_vcfheader();

                // std::cout << "cpp: vcfgz_reader::readsites() : gonna loop" << std::endl;
                for (i = 0; i < n; i++){
                    // std::cout << "cpp: vcfgz_reader::readsites() : looping i=" << i << std::endl;
                    // if (i > 100){
                    //     // std::cout << "cpp: vcfgz_reader::readsites() : i is too large! breaking the loop" << std::endl;
                    //     break;
                    // }
                    fill_in_CHR_POS_VID_from_line(
                        CHR_arr,
                        POS_arr,
                        VID_arr,
                        i
                    );
                    // std::cout << "cpp: vcfgz_reader::readsites() : looping i=" << i << "ran fill_in()" << std::endl;
                    this->table_row_i++;
                    this->line_len = readline_arr(this->ref_panel, this->cache_line);
                }
            } catch (gzReadPastEOF& e) {
                // std::cout << "cpp: vcfgz_reader::readsites() : end of file! Gonna close the FD" << std::endl;
                this->close();
            }
            // std::cout << "cpp: vcfgz_reader::readsites() : after the loop" << std::endl;

            return this->table_row_i;

        }

        const char* get_table_header(){
            return this->table_header;
        }

        int get_table_header_len(){
            return this->table_header_len;
        }
};


extern "C" {
    vcfgz_reader* vcfgz_reader_new(const char* path){
        return new vcfgz_reader(path);
    }

    long long int get_total_haploids(vcfgz_reader *reader){
        // std::cout << "reader->total_haploids = " << reader->total_haploids << std::endl;
        return reader->total_haploids;
    }

    int readlines(vcfgz_reader *reader, int8_t* table, long long int lines_n){
        return reader->readlines(table, lines_n);
    }

    int readsites(
        vcfgz_reader *reader,
        void* CHR_arr,
        void* POS_arr,
        void* VID_arr,
        long long int lines_n
    ){
        char* CHR_arr_c = static_cast<char*>(CHR_arr);
        char* POS_arr_c = static_cast<char*>(POS_arr);
        char* VID_arr_c = static_cast<char*>(VID_arr);
        return reader->readsites(CHR_arr_c, POS_arr_c, VID_arr_c, lines_n);
    }

    int readcolumns(
        vcfgz_reader *reader,
        const long long int n,
        void* const CHR_arr = nullptr,
        void* const POS_arr = nullptr,
        void* const ID_arr = nullptr,
        void* const REF_arr = nullptr,
        void* const ALT_arr = nullptr,
        void* const QUAL_arr = nullptr,
        void* const FILTER_arr = nullptr,
        void* const INFO_arr = nullptr,
        void* const FORMAT_arr = nullptr,
        void* const genotypes_arr = nullptr,
        const long long int total_haploids = 0)
    {
        // if (CHR_arr != nullptr)
        //     std::cout << "CHR is not a nullptr" << "\n";
        // if (POS_arr != nullptr)
        //     std::cout << "POS is not a nullptr" << "\n";
        // if (ID_arr != nullptr)
        //     std::cout << "ID is not a nullptr" << "\n";
        // if (REF_arr != nullptr)
        //     std::cout << "REF is not a nullptr" << "\n";
        // if (ALT_arr != nullptr)
        //     std::cout << "ALT is not a nullptr" << "\n";
        // if (QUAL_arr != nullptr)
        //     std::cout << "QUAL is not a nullptr" << "\n";
        // if (FILTER_arr != nullptr)
        //     std::cout << "FILTER is not a nullptr" << "\n";
        // if (INFO_arr != nullptr)
        //     std::cout << "INFO is not a nullptr" << "\n";
        // if (FORMAT_arr != nullptr)
        //     std::cout << "FORMAT is not a nullptr" << "\n";
        // if (genotypes_arr != nullptr)
        //     std::cout << "genotypes is not a nullptr" << "\n";
        return reader->readcolumns(
            n,
            static_cast<char*>(CHR_arr),
            static_cast<char*>(POS_arr),
            static_cast<char*>(ID_arr),
            static_cast<char*>(REF_arr),
            static_cast<char*>(ALT_arr),
            static_cast<char*>(QUAL_arr),
            static_cast<char*>(FILTER_arr),
            static_cast<char*>(INFO_arr),
            static_cast<char*>(FORMAT_arr),
            static_cast<int8_t*>(genotypes_arr),
            total_haploids
        );
    }

    const char* get_table_header(vcfgz_reader *reader){
        return reader->table_header;
    }


    // int fill_table_header(vcfgz_reader *reader, char *cache_line_copy){
    //     for (int i=0; i<reader->table_header_len; i++)
    //         cache_line_copy[i] = reader->table_header[i];
    //     return reader->table_header_len;
    // }
    int fill_table_header(vcfgz_reader *reader, char *cache_line_copy){
        for (int i=0; i<reader->table_header_len; i++)
            cache_line_copy[i] = reader->table_header_copy[i];
        return reader->table_header_len;
    }

    int get_table_header_len(vcfgz_reader *reader){
        return reader->table_header_len;
    }

    int get_cache_line_len(vcfgz_reader *reader){
        return reader->CACHE_LINE_LEN_prop;
    }

    void close_(vcfgz_reader *reader){
        reader->close();
    }
}


// some unit test
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
