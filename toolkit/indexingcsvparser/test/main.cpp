#define INDCSVP_REPLACEDECIMALCOMMA
#include "indexingcsvparser/indexingcsvparser.h"

#define CATCH_CONFIG_MAIN
#include "catch2/catch.hpp"

using namespace std;
using namespace indcsvp;

enum class Parser {ShouldSucceed, ShouldFail};
enum class TrimOption {NoTrim, Trim};

struct TestFile
{
    char separator;
    char decimalSeparator;
    size_t columnHeaderRowIndex;
    TrimOption trimSpaces;
    string file;
};


static vector<TestFile> intFiles { {',', '\0', 2, TrimOption::NoTrim, "testdata_int_comma_h3r11c3_lf.csv"},
                                   {',', '\0', 2, TrimOption::NoTrim, "testdata_int_comma_h3r11c3_crlf.csv"},
                                   {';', '\0', 2, TrimOption::NoTrim, "testdata_int_semicolon_h3r11c3_lf.csv"},
                                   {';', '\0', 2, TrimOption::NoTrim, "testdata_int_semicolon_h3r11c3_crlf.csv"},
                                   {',', '\0', 2, TrimOption::Trim,  "testdata_int_comma-space_h3r11c3_lf.csv"},};

static vector<TestFile> realFiles {{',', '.', 2, TrimOption::NoTrim, "testdata_real_comma_h3r9c2_lf.csv"},
                                   {',', '.', 2, TrimOption::NoTrim, "testdata_real_comma_h3r9c2_crlf.csv"},
                                   {';', ',', 2, TrimOption::NoTrim, "testdata_real_semicolon_decimalcomma_h3r9c2_lf.csv"},
                                   {';', ',', 2, TrimOption::NoTrim, "testdata_real_semicolon_decimalcomma_h3r9c2_crlf.csv"}};

static vector<TestFile> headerFiles {{',', '.', 2, TrimOption::NoTrim, "testdata_header_ascomment_h4.csv"},
                                     {',', '.', 2, TrimOption::NoTrim, "testdata_header_belowcomment_h3.csv"},
                                     {',', '.', 2, TrimOption::Trim,   "testdata_header_belowcomment_indented_h3.csv"},
                                     {',', '.', 2, TrimOption::NoTrim, "testdata_header_belowcomment_indented_h3.csv"},};


class TestFileGenerator : public Catch::Generators::IGenerator<TestFile>
{
    vector<TestFile> mFiles;
    size_t mIdx = 0;

public:
    TestFileGenerator(vector<TestFile> files) {
        append(std::move(files));
    }

    TestFileGenerator(vector<TestFile> files, vector<TestFile> morefiles... ) {
        append(std::move(files), std::move(morefiles));
    }

    const TestFile& get() const override {
        return mFiles[mIdx];
    }

    bool next() override {
        if (mIdx+1 < mFiles.size()) {
            mIdx++;
            return true;
        }
        return false;
    }
protected:
    void append(vector<TestFile>&& files) {
        for (auto&& f : files) {
            mFiles.emplace_back(std::move(f));
        }
    }

    void append(vector<TestFile>&& files, vector<TestFile>&& morefiles...){
        append(std::forward<vector<TestFile>>(files));
        append(std::forward<vector<TestFile>>(morefiles));
    }
};

class IndexingCSVParserSpy : public IndexingCSVParser
{
public:
    IndexingCSVParserSpy() : IndexingCSVParser() {}
    bool fileIsOpen() const {
        return mpFile != nullptr;
    }

    bool fileIsAtBeginning() const {
        return fileIsOpen() && ftell(mpFile) == 0;
    }

    bool fileIsAtEnd() const {
        return fileIsOpen() && feof(mpFile) != 0;
    }
};


template<typename T>
typename std::enable_if<std::is_floating_point<T>::value, bool>::type compare(T value1, T value2) {
    return Approx(value1) == value2;
}

template<typename T>
typename std::enable_if<!std::is_floating_point<T>::value, bool>::type compare(T value1, T value2) {
    return value1 == value2;
}

TrimSpaceOption convertTrimArg(TrimOption trim) {
    return trim==TrimOption::Trim ? TrimLeadingTrailingSpace : NoTrim;
}

template <typename T>
class CheckIndexedPosMatcher : public Catch::MatcherBase<T>
{
    IndexingCSVParserSpy* mpCsvp;
    size_t mRow;
    size_t mCol;
    Parser mParserExpectation;
    TrimOption mTrimWhitespaces;
    mutable bool mParseOK;
    mutable T mValue;

public:
    CheckIndexedPosMatcher(IndexingCSVParserSpy* pCsvp, size_t row, size_t col, Parser parserExpectation, TrimOption trim) {
        mpCsvp = pCsvp;
        mRow = row;
        mCol = col;
        mParserExpectation = parserExpectation;
        mTrimWhitespaces = trim;
        mParseOK = false;
    }

    virtual bool match( const T& expectedValue ) const override {
        mValue = mpCsvp->getIndexedPosAs<T>(mRow, mCol, mParseOK, convertTrimArg(mTrimWhitespaces));
        if (mParserExpectation == Parser::ShouldFail) {
            return !mParseOK;
        }
        return compare(mValue, expectedValue);
    }

    virtual std::string describe() const override {
        std::ostringstream ss;
        if ((mParserExpectation == Parser::ShouldFail) && mParseOK) {
            ss << "Parsing should have failed at row: " << mRow << " col: " << mCol
               << ", Total num rows: " << mpCsvp->numRows() << " cols: " << mpCsvp->numCols();
        }
        else if (mParseOK) {
            ss << "!= " << mValue << " at row: " << mRow << " col: " << mCol;
        }
        else {
            ss << "could not be verified due to parsing failure at row: " << mRow << " col: " << mCol
               << ", Total num rows: " << mpCsvp->numRows() << " cols: " << mpCsvp->numCols();
        }
        return ss.str();
    }
};

template <typename VectorT>
class IndexedRowOrColMatcher : public Catch::MatcherBase<VectorT>
{
    IndexingCSVParserSpy* mpCsvp;
    size_t mRowOrCol;
    Direction mDirection;
    Parser mParserExpectation;
    TrimOption mTrimWhitespaces;
    mutable bool mParseOK;
    mutable VectorT mValues;

public:
    IndexedRowOrColMatcher(IndexingCSVParserSpy* pCsvp, size_t roc, Direction direction, Parser parserexpectation, TrimOption trim) {
        mpCsvp = pCsvp;
        mRowOrCol = roc;
        mDirection = direction;
        mParserExpectation = parserexpectation;
        mTrimWhitespaces = trim;
        mParseOK = false;
    }

    virtual bool match( const VectorT& expectedValues ) const override {
        if (mDirection == Direction::Row) {
            mParseOK = mpCsvp->getIndexedRowAs<typename VectorT::value_type>(mRowOrCol, mValues, convertTrimArg(mTrimWhitespaces));
        }
        else {
            mParseOK = mpCsvp->getIndexedColumnAs<typename VectorT::value_type>(mRowOrCol, mValues, convertTrimArg(mTrimWhitespaces));
        }
        if (mParserExpectation == Parser::ShouldFail) {
            return !mParseOK;
        }
        return mValues == expectedValues;
    }

    virtual std::string describe() const override {
        std::ostringstream ss;
        string rorc = (mDirection == Direction::Row ? " at row: " : " at column: ");
        if ((mParserExpectation == Parser::ShouldFail) && mParseOK) {
            ss << "Parsing should have failed" << rorc << mRowOrCol;
        }
        else if (mParseOK) {
            ss << "does not match "  << rorc << mRowOrCol;
        }
        else {
            ss << "could not be verified due to parsing failure " << rorc << mRowOrCol
               << ", Total num rows: " << mpCsvp->numRows() << " cols: " << mpCsvp->numCols();
        }
        return ss.str();
    }
};

template <typename VectorT>
class IndexedRowOrColRangeMatcher : public Catch::MatcherBase<VectorT>
{
    IndexingCSVParserSpy* mpCsvp;
    size_t mRowOrCol;
    size_t mStart;
    size_t mEnd;
    Direction mDirection;
    Parser mParserExpectation;
    TrimOption mTrimWhitespaces;
    mutable bool mParseOK;
    mutable VectorT mValues;

public:
    IndexedRowOrColRangeMatcher(IndexingCSVParserSpy* pCsvp, size_t roc, Direction direction, size_t start, size_t end, Parser parserexpectation, TrimOption trim) {
        mpCsvp = pCsvp;
        mRowOrCol = roc;
        mStart = start;
        mEnd = end;
        mDirection = direction;
        mParserExpectation = parserexpectation;
        mTrimWhitespaces = trim;
        mParseOK = false;
    }

    virtual bool match( const VectorT& expectedValues ) const override {
        if (mDirection == Direction::Row) {
            mParseOK = mpCsvp->getIndexedRowColumnRangeAs<typename VectorT::value_type>(mRowOrCol, mStart, mEnd, mValues, convertTrimArg(mTrimWhitespaces));
        }
        else {
            mParseOK = mpCsvp->getIndexedColumnRowRangeAs<typename VectorT::value_type>(mRowOrCol, mStart, mEnd, mValues, convertTrimArg(mTrimWhitespaces));
        }
        if (mParserExpectation == Parser::ShouldFail) {
            return !mParseOK;
        }
        return mValues == expectedValues;
    }

    virtual std::string describe() const override {
        std::ostringstream ss;
        string rorc = (mDirection == Direction::Row ? " at row: " : " at column: ");
        if ((mParserExpectation == Parser::ShouldFail) && mParseOK) {
            ss << "Parsing should have failed" << rorc << mRowOrCol;
        }
        else if (mParseOK) {
            ss << "does not match "  << rorc << mRowOrCol;
        }
        else {
            ss << "could not be verified due to parsing failure " << rorc << mRowOrCol
               << ", Total num rows: " << mpCsvp->numRows() << " cols: " << mpCsvp->numCols();
        }
        return ss.str();
    }
};




template <typename T>
CheckIndexedPosMatcher<T> AtIndex(IndexingCSVParserSpy* pCsvp, size_t row, size_t col, TrimOption trim) {
    return CheckIndexedPosMatcher<T>(pCsvp, row, col, Parser::ShouldSucceed, trim);
}

CheckIndexedPosMatcher<string> AtIndexParseFail(IndexingCSVParserSpy* pCsvp, size_t row, size_t col, TrimOption trim) {
    return CheckIndexedPosMatcher<string>(pCsvp, row, col, Parser::ShouldFail, trim);
}

template <typename VectorT>
IndexedRowOrColMatcher<VectorT> AtRow(IndexingCSVParserSpy* pCsvp, size_t row, TrimOption trim) {
    return IndexedRowOrColMatcher<VectorT>(pCsvp, row, Direction::Row, Parser::ShouldSucceed, trim);
}

IndexedRowOrColMatcher<vector<string>> AtRowParseFail(IndexingCSVParserSpy* pCsvp, size_t row, TrimOption trim) {
    return IndexedRowOrColMatcher<vector<string>>(pCsvp, row, Direction::Row, Parser::ShouldFail, trim);
}

template <typename VectorT>
IndexedRowOrColMatcher<VectorT> AtColumn(IndexingCSVParserSpy* pCsvp, size_t col, TrimOption trim) {
    return IndexedRowOrColMatcher<VectorT>(pCsvp, col, Direction::Column, Parser::ShouldSucceed, trim);
}

IndexedRowOrColMatcher<vector<string>> AtColumnParseFail(IndexingCSVParserSpy* pCsvp, size_t col, TrimOption trim) {
    return IndexedRowOrColMatcher<vector<string>>(pCsvp, col, Direction::Column, Parser::ShouldFail, trim);
}

template <typename VectorT>
IndexedRowOrColRangeMatcher<VectorT> AtRowColumnRange(IndexingCSVParserSpy* pCsvp, size_t row, size_t colstart, size_t colnum, TrimOption trim) {
    return IndexedRowOrColRangeMatcher<VectorT>(pCsvp, row, Direction::Row, colstart, colnum, Parser::ShouldSucceed, trim);
}

IndexedRowOrColRangeMatcher<vector<string>> AtRowColumnRangeParseFail(IndexingCSVParserSpy* pCsvp, size_t row, size_t colstart, size_t colnum, TrimOption trim) {
    return IndexedRowOrColRangeMatcher<vector<string>>(pCsvp, row, Direction::Row, colstart, colnum, Parser::ShouldFail, trim);
}

template <typename VectorT>
IndexedRowOrColRangeMatcher<VectorT> AtColumnRowRange(IndexingCSVParserSpy* pCsvp, size_t col, size_t rowstart, size_t rownum, TrimOption trim) {
    return IndexedRowOrColRangeMatcher<VectorT>(pCsvp, col, Direction::Column, rowstart, rownum, Parser::ShouldSucceed, trim);
}

IndexedRowOrColRangeMatcher<vector<string>> AtColumnRowRangeParseFail(IndexingCSVParserSpy* pCsvp, size_t col, size_t rowstart, size_t rownum, TrimOption trim) {
    return IndexedRowOrColRangeMatcher<vector<string>>(pCsvp, col, Direction::Column, rowstart, rownum, Parser::ShouldFail, trim);
}



TEST_CASE("Set parse parameters") {
    IndexingCSVParserSpy csvp;

    SECTION("Separator char") {
        CHECK(csvp.getSeparatorChar() == ',');
        csvp.setSeparatorChar(';');
        REQUIRE(csvp.getSeparatorChar() == ';');
    }

    SECTION("Comment char") {
        CHECK(csvp.getCommentChar() == '\0');
        csvp.setCommentChar('#');
        REQUIRE(csvp.getCommentChar() == '#');
    }

    SECTION("Number of lines to skip") {
        CHECK(csvp.getNumLinesToSkip() == 0);
        csvp.setNumLinesToSkip(3);
        REQUIRE(csvp.getNumLinesToSkip() == 3);
    }

    SECTION("Header setting") {
        auto header = csvp.getHeaderSetting();
        REQUIRE_FALSE(header.isValid());
        csvp.setHeaderInfo(Row, 5);
        header = csvp.getHeaderSetting();
        CHECK(header.isValid());
        CHECK(header.direction() == Row);
        CHECK(header.rowOrColumn() == 5);
        csvp.setHeaderInfo(Column, 6);
        header = csvp.getHeaderSetting();
        CHECK(header.isValid());
        CHECK(header.direction() == Column);
        CHECK(header.rowOrColumn() == 6);
    }
}

TEST_CASE("File operations") {
    IndexingCSVParserSpy csvp;
    csvp.openFile(intFiles.front().file.c_str());
    INFO(intFiles.front().file.c_str());
    REQUIRE(csvp.fileIsOpen());
    REQUIRE(csvp.fileIsAtBeginning());
    csvp.indexFile();
    REQUIRE(csvp.fileIsAtEnd());
    csvp.rewindFile();
    REQUIRE(csvp.fileIsAtBeginning());
    csvp.closeFile();
    REQUIRE_FALSE(csvp.fileIsOpen());
}

TEST_CASE("External file operations") {
    IndexingCSVParserSpy csvp;
    FILE* pTempfile;
#if defined(_WIN32)
    char tmpfilebuff[L_tmpnam];
    errno_t rc = tmpnam_s(tmpfilebuff, L_tmpnam);
    REQUIRE(rc == 0);
    //INFO(tmpfilebuff);
    pTempfile = fopen(tmpfilebuff, "w+b");
#else
    char filetemplate[20] {"indscvp-temp-XXXXXX"};
    int fd = mkstemp(filetemplate);
    pTempfile = fdopen(fd, "r+");
#endif
    REQUIRE(pTempfile != 0);
    fputs("1,2,3", pTempfile);
    rewind(pTempfile);

    csvp.takeOwnershipOfFile(pTempfile);
    REQUIRE(csvp.fileIsOpen());
    REQUIRE(csvp.fileIsAtBeginning());
    csvp.indexFile();
    REQUIRE(csvp.numRows() == 1);
    REQUIRE(csvp.numCols() == 3);
    REQUIRE(csvp.fileIsAtEnd());
    csvp.rewindFile();
    REQUIRE(csvp.fileIsAtBeginning());
    csvp.closeFile();
    REQUIRE_FALSE(csvp.fileIsOpen());
}

TEST_CASE("Parse integer data") {
    auto testfile = GENERATE(GeneratorWrapper<TestFile>(std::unique_ptr<IGenerator<TestFile>>(new TestFileGenerator(intFiles))));

    IndexingCSVParserSpy csvp;
    csvp.setSeparatorChar(testfile.separator);
    csvp.openFile(testfile.file.c_str());
    csvp.setCommentChar('#');
    csvp.setHeaderInfo(Row, testfile.columnHeaderRowIndex);
    INFO(testfile.file.c_str());
    REQUIRE(csvp.fileIsOpen());

    csvp.indexFile();

    SECTION("Check lines and columns") {
        REQUIRE(csvp.numRows() == 11);
        REQUIRE(csvp.numCols() == 3);
        CHECK(csvp.allRowsHaveSameNumCols());
        size_t mincols, maxcols;
        csvp.minMaxNumCols(mincols, maxcols);
        CHECK(mincols == csvp.numCols());
        CHECK(maxcols == csvp.numCols());
    }

    SECTION("Check header") {
        auto header = csvp.header();
        decltype(header) expected {"v", "v squared", "v multiplied with 2"};
        CHECK(expected == header);
    }

    SECTION("Extract individual values") {
        CHECK_THAT("0", AtIndex<string>(&csvp, 0, 0, testfile.trimSpaces));
        CHECK_THAT("9", AtIndex<string>(&csvp, 3, 1, testfile.trimSpaces));
        CHECK_THAT("6", AtIndex<string>(&csvp, 3, 2, testfile.trimSpaces));

        CHECK_THAT(0, AtIndex<int>(&csvp, 0, 0, testfile.trimSpaces));
        CHECK_THAT(9, AtIndex<int>(&csvp, 3, 1, testfile.trimSpaces));
        CHECK_THAT(6, AtIndex<int>(&csvp, 3, 2, testfile.trimSpaces));

        CHECK_THAT(0.0, AtIndex<double>(&csvp, 0, 0, testfile.trimSpaces));
        CHECK_THAT(9.0, AtIndex<double>(&csvp, 3, 1, testfile.trimSpaces));
        CHECK_THAT(6.0, AtIndex<double>(&csvp, 3, 2, testfile.trimSpaces));
    }

    SECTION("Extract row") {
        vector<string> expected = {"2", "4", "4"};
        CHECK_THAT(expected, AtRow<decltype(expected)>(&csvp, 2, testfile.trimSpaces));
        vector<float> expectedf = {2.0, 4.0, 4.0};
        CHECK_THAT(expectedf, AtRow<decltype(expectedf)>(&csvp, 2, testfile.trimSpaces));
    }

    SECTION("Extract row column range") {
        vector<string> expected = {"9", "6"};
        CHECK_THAT(expected, AtRowColumnRange<vector<string>>(&csvp, 3, 1, 2, testfile.trimSpaces));
    }

    SECTION("Extract column") {
        vector<string> expected = {"0", "1", "4", "9", "16", "25", "36", "49", "64", "81", "100" };
        CHECK_THAT(expected, AtColumn<vector<string>>(&csvp, 1, testfile.trimSpaces));
    }

    SECTION("Extract column row range") {
        vector<string> expected = {"16", "25", "36"};
        CHECK_THAT(expected, AtColumnRowRange<decltype(expected)>(&csvp, 1, 4, 3, testfile.trimSpaces));
        vector<unsigned long int> expecteduli = {16, 25, 36};
        CHECK_THAT(expecteduli, AtColumnRowRange<decltype(expecteduli)>(&csvp, 1, 4, 3, testfile.trimSpaces));
    }

    csvp.closeFile();
}

TEST_CASE("Parse real data") {
    auto testfile = GENERATE(GeneratorWrapper<TestFile>(std::unique_ptr<IGenerator<TestFile>>(new TestFileGenerator(realFiles))));

    IndexingCSVParserSpy csvp;
    csvp.setSeparatorChar(testfile.separator);
    string ds {testfile.decimalSeparator};
    csvp.openFile(testfile.file.c_str());
    csvp.setCommentChar('#');
    csvp.setHeaderInfo(Row, testfile.columnHeaderRowIndex);
    INFO(testfile.file.c_str());
    REQUIRE(csvp.fileIsOpen());

    csvp.indexFile();

    SECTION("Check lines and columns") {
        REQUIRE(csvp.numRows() == 9);
        REQUIRE(csvp.numCols() == 2 );
        CHECK(csvp.allRowsHaveSameNumCols());
        size_t mincols, maxcols;
        csvp.minMaxNumCols(mincols, maxcols);
        CHECK(mincols == csvp.numCols());
        CHECK(maxcols == csvp.numCols());
    }

    SECTION("Check header") {
        auto header = csvp.header();
        decltype(header) expected {"x", "x+0"+ds+"5"};
        CHECK(expected == header);
    }

    SECTION("Extract individual values") {
        CHECK_THAT("1e-1", AtIndex<string>(&csvp, 0, 0, testfile.trimSpaces));
        CHECK_THAT("0"+ds+"15", AtIndex<string>(&csvp, 0, 1, testfile.trimSpaces));
        CHECK_THAT("5"+ds+"555", AtIndex<string>(&csvp, 5, 0, testfile.trimSpaces));
        CHECK_THAT("1"+ds+"05e+1", AtIndex<string>(&csvp, 8, 1, testfile.trimSpaces));

        CHECK_THAT(8, AtIndex<int>(&csvp, 6, 1, testfile.trimSpaces));

        CHECK_THAT(0.1, AtIndex<double>(&csvp, 0, 0, testfile.trimSpaces));
        CHECK_THAT(0.15, AtIndex<double>(&csvp, 0, 1, testfile.trimSpaces));
        CHECK_THAT(6.055, AtIndex<double>(&csvp, 5, 1, testfile.trimSpaces));
        CHECK_THAT(10.50, AtIndex<double>(&csvp, 8, 1, testfile.trimSpaces));
    }

    SECTION("Extract row") {
        vector<string> expected = {"1"+ds, "1"+ds+"50"};
        CHECK_THAT(expected, AtRow<decltype(expected)>(&csvp, 2, testfile.trimSpaces));
        vector<float> expectedf = {1, 1.5};
        CHECK_THAT(expectedf, AtRow<decltype(expectedf)>(&csvp, 2, testfile.trimSpaces));
    }

    SECTION("Extract row column range") {
        vector<string> expected = {"1"+ds+"05e+1"};
        CHECK_THAT(expected, AtRowColumnRange<vector<string>>(&csvp, 8, 1, 1, testfile.trimSpaces));
    }

    SECTION("Extract column") {
        vector<string> expected = {"1e-1", "0"+ds+"00", "1"+ds, "2"+ds+"20", "4"+ds+"44", "5"+ds+"555", "8", "0"+ds+"9e+1", "1e+1"};
        CHECK_THAT(expected, AtColumn<vector<string>>(&csvp, 0, testfile.trimSpaces));
    }

    SECTION("Extract column row range") {
        vector<string> expected = {"0"+ds+"95E+1", "1"+ds+"05e+1"};
        CHECK_THAT(expected, AtColumnRowRange<decltype(expected)>(&csvp, 1, 7, 2, testfile.trimSpaces));
        vector<double> expectedd = {9.5, 10.5};
        CHECK_THAT(expectedd, AtColumnRowRange<decltype(expectedd)>(&csvp, 1, 7, 2, testfile.trimSpaces));
    }

    csvp.closeFile();
}

TEST_CASE("Check out of bounds")
{
    auto testfile = GENERATE(GeneratorWrapper<TestFile>(std::unique_ptr<IGenerator<TestFile>>(new TestFileGenerator(intFiles, realFiles))));

    IndexingCSVParserSpy csvp;
    csvp.setSeparatorChar(testfile.separator);
    csvp.openFile(testfile.file.c_str());
    csvp.setCommentChar('#');
    INFO(testfile.file.c_str());
    REQUIRE(csvp.fileIsOpen());

    csvp.indexFile();

    CHECK_THAT("", AtIndexParseFail(&csvp, csvp.numRows(), 0, testfile.trimSpaces));
    CHECK_THAT("", AtIndexParseFail(&csvp, 0, csvp.numCols(), testfile.trimSpaces));
    CHECK_THAT("", AtIndexParseFail(&csvp, csvp.numRows(), csvp.numCols(), testfile.trimSpaces));

    vector<string> expected {"nothing"};
    CHECK_THAT(expected, AtRowParseFail(&csvp, csvp.numRows(), testfile.trimSpaces));
    CHECK_THAT(expected, AtColumnParseFail(&csvp, csvp.numCols(), testfile.trimSpaces));

    csvp.closeFile();
}

TEST_CASE("Check header")
{
    auto testfile = GENERATE(GeneratorWrapper<TestFile>(std::unique_ptr<IGenerator<TestFile>>(new TestFileGenerator(headerFiles))));

    IndexingCSVParserSpy csvp;
    csvp.setSeparatorChar(testfile.separator);
    csvp.openFile(testfile.file.c_str());
    csvp.setCommentChar('#');
    csvp.setHeaderInfo(Row, testfile.columnHeaderRowIndex);
    INFO(testfile.file.c_str());
    REQUIRE(csvp.fileIsOpen());

    csvp.indexFile();

    auto header = csvp.header();
    decltype(header) expected {"v", "v squared", "v multiplied with 2"};
    CHECK(expected == header);
}
