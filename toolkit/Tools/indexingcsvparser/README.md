# IndexingCSVParser

## Introduction

This is a simple CSV (Comma Separated Value) ASCII text file parser implemented as a C++ class.
It can operate in two modes.

1. Parse all data in the input file line by line.
2. Index the file to make it possible to extract and parse individual rows or columns

Method 1 is suitable if you want to parse all data in the file.<br>
Method 2 is suitable if you have a large data sets and only want to extract (and parse) one or a few specific columns or rows.

## Features
- Cross platform (only tested on Linux and Windows (MinGW))
- Supports both LF and CRLF end-of-line characters.
- You can specify a "comment" character, to make the parser ignore initial lines beginning with that character.
- You can specify the number of initial lines to skip.
- You can chose what separator character to use.
- The parser can also automatically detect which separator character to use (based on supplied alternatives)
- Indexing allows you to count the number of rows and columns before extracting data. This allows you to reserve memory before extraction and parsing begins.
- You can extract (and parse) individual rows or columns (if file has been indexed).
- Template functions allow you to extract data and parse it as a specific type directly, instead of first extracting as string and then converting in an additional step.
- A header row or column can be specified and extracted.
- Leading and trailing white-spaces can be trimmed (optional) while parsing data as strings.

## Current Limitations
- Error handling is limited, it is assumed that no read errors on the file occur.
- The data file must be "well formed" the same separator character must be used in the entire file.
- The parser does not yet support "enclosure". A way to enclose text containing the separator character without it being interpreted as a separator.
- Supports only CRLF or LF EOL characters.
- Only supports ASCII input (no Unicode support).

## License
This code is released under the Simplified BSD license. See the LICENSE file for details

## Types
Type conversion is done using specialized template converter functions, see IndexingCSVParserImpl.hpp for details.
Support for new types need a new template specialization for that type. 
Currently the following type conversions are implemented:
- float
- double
- int
- unsigned int
- long int
- unsigned long int
- std::string

## Configuration
If you define INDCSVP_REPLACEDECIMALCOMMA before including the IndexingCSVParser.h header, the library will automatically replace decimal mark ',' with '.' during conversion to double values. This makes it possible to parse numeric CSV files exported from some programs under some specific locale settings, even if your application uses for example the C locale.

## Build
The library is written in plain C++ 98 (tests require C++ 11) and uses CMake for the build system to produce a static library. You can also choose to include the source code files directly into your project. Just copy the include and src directories.

## Usage Example 
See the example/main.cpp file for usage example, and read the Doxygen comments in the code (cpp files).

