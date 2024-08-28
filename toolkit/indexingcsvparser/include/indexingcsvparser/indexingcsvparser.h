//!
//! @file   indexingcsvparser.h
//! @author Peter Nordin
//! @date   2015-02-03
//!
//! @brief Contains the definition of the IndexingCSVParser
//!

#ifndef INDEXINGCSVPARSER_H
#define INDEXINGCSVPARSER_H

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

//! @brief The Indexing CSV Parser namespace
namespace indcsvp
{

enum Direction {Row, Column};
enum TrimSpaceOption {NoTrim, TrimLeadingTrailingSpace};

class HeaderSetting {

public:
    HeaderSetting();
    void setHeaderInfo(Direction direction, size_t rowOrCol);
    bool isValid() const;
    Direction direction() const;
    size_t rowOrColumn() const;

protected:
    size_t mHeaderIndex;
    Direction mDirection;
};

//! @brief The Indexing CSV Parser Class
class IndexingCSVParser
{
public:
    IndexingCSVParser();

    // ----- Configuration methods -----
    void setHeaderInfo(Direction direction, size_t rowOrColumn);
    void setSeparatorChar(char sep);
    void setCommentChar(char com);
    void setNumLinesToSkip(size_t num);
    char autoSetSeparatorChar(const std::vector<char> &rAlternatives);
    HeaderSetting getHeaderSetting() const;
    char getSeparatorChar() const;
    char getCommentChar() const;
    size_t getNumLinesToSkip() const;

    // ----- File methods -----
    bool openFile(const char* filePath);
    void takeOwnershipOfFile(FILE* pExternalFile);
    void closeFile();
    void rewindFile();
    void readUntilData();
    void skipNumLines(size_t num);

    // ----- Indexing access methods -----
    void indexFile();

    size_t numRows() const;
    size_t numCols(size_t row=0) const;
    bool allRowsHaveSameNumCols() const;
    void minMaxNumCols(size_t &rMin, size_t &rMax);
    const std::vector<std::string>& header() const;

    bool getIndexedColumn(const size_t col, std::vector<std::string> &rData, TrimSpaceOption trim=NoTrim);
    bool getIndexedRow(const size_t row, std::vector<std::string> &rData, TrimSpaceOption trim=NoTrim);
    std::string getIndexedPos(const size_t row, const size_t col, bool &rParseOK, TrimSpaceOption trim=NoTrim);

    template <typename T> bool getIndexedColumnAs(const size_t col, std::vector<T> &rData, TrimSpaceOption trim=NoTrim);
    template <typename T> bool getIndexedColumnRowRangeAs(const size_t col, const size_t startRow, const size_t numRows, std::vector<T> &rData, TrimSpaceOption trim=NoTrim);
    template <typename T> bool getIndexedRowAs(const size_t row, std::vector<T> &rData, TrimSpaceOption trim=NoTrim);
    template <typename T> bool getIndexedRowColumnRangeAs(const size_t row, const size_t startCol, const size_t numCols, std::vector<T> &rData, TrimSpaceOption trim=NoTrim);
    template <typename T> T getIndexedPosAs(const size_t row, const size_t col, bool &rParseOK, TrimSpaceOption trim=NoTrim);

    // ----- Non-indexing access methods -----
    bool getRow(std::vector<std::string> &rData, TrimSpaceOption trim=NoTrim);
    template <typename T> bool getRowAs(std::vector<T> &rData, TrimSpaceOption trim=NoTrim);
    bool hasMoreDataRows();

protected:
    bool getHeaderRow(std::vector<std::string> &rData);

    FILE *mpFile;           //!< @brief The internal file pointer
    char mSeparatorChar;    //!< @brief The chosen separator character
    char mCommentChar;      //!< @brief The chosen comment character
    size_t mNumSkipLines;   //!< @brief The initial lines to skip
    std::vector< std::vector<size_t> > mSeparatorMatrix; //!< @brief The index of separators
    std::vector< std::string > mHeader;
    HeaderSetting mHeaderSetting;
};






// ============================================================
// Header implementation below
// ============================================================

inline void replaceDecimalComma(char* pBuff)
{
    size_t i=0;
    while (pBuff[i] != '\0')
    {
        if (pBuff[i] == ',')
        {
            pBuff[i] = '.';
            break;
        }
        ++i;
    }
}

//! @brief Character buffer help class, with automatic memory deallocation and smart reallocation
class CharBuffer
{
public:
    CharBuffer() : mpBuffer(0), mAllocatedSize(0), mContentSize(0) {}
    CharBuffer(size_t size) : mpBuffer(0), mAllocatedSize(0), mContentSize(0) {setContentSize(size);}
    ~CharBuffer()
    {
        if (mpBuffer)
        {
            delete[] mpBuffer;
        }
    }

    //! @brief Set the buffer size for intended content, excluding null-terminator space
    inline bool setContentSize(size_t size)
    {
        bool alloc_ok = resizeBuffer(size+1);
        if (alloc_ok) {
            mContentSize = size;
        }
        return alloc_ok;
    }

    //! @brief Returns the current buffer content size (excluding null terminator)
    inline size_t contentSize()
    {
        return mContentSize;
    }

    //! @brief Returns the current allocated buffer size
    inline size_t bufferSize()
    {
        return mAllocatedSize;
    }

    //! @brief Returns the actual character buffer
    inline char* buff()
    {
        return mpBuffer;
    }

    //! @brief Returns a string constructed from the character buffer
    std::string str(TrimSpaceOption trim=NoTrim) const
    {
        if (trim==TrimLeadingTrailingSpace && mContentSize>1) {
            return trimmed_str();
        }
        else {
            return std::string(mpBuffer);
        }
    }

    //! @brief The default converter template function
    //! @details This function will always fail, template specialization for each type are required
    //! @tparam T The type that we want to interpret the contests of pBuffer as.
    //! @param[out] rIsOK Reference to bool flag telling you if parsing completed successfully
    //! @param[in] trim Whether to trim leading and trailing spaces from data
    //! @returns Type default constructed value;
    template <typename T>
    T getAs(bool &rIsOK, TrimSpaceOption trim=NoTrim)
    {
        static_cast<void>(trim);
        rIsOK = false;
        return T();
    }

protected:
    std::string trimmed_str() const
    {
        char* start = mpBuffer;
        size_t count = mContentSize;
        for (size_t i=0; i<mContentSize; ++i) {
            if (isspace(mpBuffer[i])) {
                start = &mpBuffer[i+1];
                --count;
            }
            else {
                break;
            }
        }
        for (size_t ri=mContentSize-1; ri>0; --ri) {
            if (isspace(mpBuffer[ri])) {
                --count;
            }
            else {
                break;
            }
        }
        return std::string(start, count);
    }

    //! @brief Reallocate the buffer memory (but only if new size is larger then before)
    //! @param[in] size The desired buffer size (the number of bytes to allocate)
    //! @returns true if reallocation was a success or if no reallocation was necessary, false if reallocation failed
    bool resizeBuffer(size_t size)
    {
        if (size > mAllocatedSize)
        {
            // Size is larger then before (reallocate memory)
            mpBuffer = static_cast<char*>(realloc(mpBuffer, size));
            if (mpBuffer)
            {
                mAllocatedSize = size;
                return true;
            }
            else
            {
                mAllocatedSize = 0;
                return false;
            }
        }
        // Lets keep the previously allocated memory as buffer (to avoid time consuming reallocation)
        return true;
    }

    char *mpBuffer;
    size_t mAllocatedSize;
    size_t mContentSize;
};

//! @brief The std::string converter specialized template function
//! @param[out] rIsOK Reference to bool flag telling you if parsing completed successfully
//! @param[in] trim Whether to trim leading and trailing spaces from data
//! @returns The contents of pBuff as a std::string
template<> inline
std::string CharBuffer::getAs<std::string>(bool &rIsOK, TrimSpaceOption trim)
{
    rIsOK = true;
    return str(trim);
}

template<> inline
double CharBuffer::getAs<double>(bool &rIsOK, TrimSpaceOption /*trim*/)
{
#ifdef INDCSVP_REPLACEDECIMALCOMMA
    replaceDecimalComma(mpBuffer);
#endif
    char *pEnd;
    double d = std::strtod(mpBuffer, &pEnd);
    rIsOK = (*pEnd == '\0') || isspace(*pEnd);
    return d;
}

template<> inline
float CharBuffer::getAs<float>(bool &rIsOK, TrimSpaceOption trim)
{
    return static_cast<float>(getAs<double>(rIsOK, trim));
}

template<> inline
unsigned long int CharBuffer::getAs<unsigned long int>(bool &rIsOK, TrimSpaceOption /*trim*/)
{
    char *pEnd;
    unsigned long int ul = strtoul(mpBuffer, &pEnd, 10); //!< @todo maybe support other bases then 10, see strtol documentation
    rIsOK = (*pEnd == '\0') || isspace(*pEnd);
    return ul;
}

template<> inline
unsigned int CharBuffer::getAs<unsigned int>(bool &rIsOK, TrimSpaceOption trim)
{
    return static_cast<unsigned int>(getAs<unsigned long int>(rIsOK, trim));
}

template<> inline
long int CharBuffer::getAs<long int>(bool &rIsOK, TrimSpaceOption /*trim*/)
{
    char *pEnd;
    long int i = strtol(mpBuffer, &pEnd, 10); //!< @todo maybe support other bases then 10, see strtol documentation
    rIsOK = (*pEnd == '\0') || isspace(*pEnd);
    return i;
}

template<> inline
int CharBuffer::getAs<int>(bool &rIsOK, TrimSpaceOption trim)
{
    return static_cast<int>(getAs<long int>(rIsOK, trim));
}


//! @brief Peek help function to peek at the next character in the file
//! @param[in] pStream The stream too peek in
//! @returns The next character (as int)
inline int peek(FILE *pStream)
{
    int c = fgetc(pStream);
    ungetc(c, pStream);
    return c;
}

template <typename T>
bool IndexingCSVParser::getIndexedColumnAs(const size_t col, std::vector<T> &rData, TrimSpaceOption trim)
{
    return IndexingCSVParser::getIndexedColumnRowRangeAs<T>(col, 0, numRows(), rData, trim);
}

template <typename T>
bool IndexingCSVParser::getIndexedColumnRowRangeAs(const size_t col, const size_t startRow, const size_t numRows, std::vector<T> &rData, TrimSpaceOption trim)
{
    // Assume all rows have same number of columns
    if (col < numCols(startRow))
    {
        // Reserve data (will only increase reserved memory if needed, not shrink)
        rData.reserve(numRows);

        // Temporary buffer object
        CharBuffer cb;

        // Loop through each row
        for (size_t r=startRow; r<startRow+numRows; ++r)
        {
            // Begin and end positions
            size_t b = mSeparatorMatrix[r][col] + size_t(col > 0);
            size_t e = mSeparatorMatrix[r][col+1];
            // Move file pointer
            std::fseek(mpFile, b, SEEK_SET);

            // Extract data
            cb.setContentSize(e-b);
            char* rc = fgets(cb.buff(), e-b+1, mpFile);
            // Push back data
            if (rc)
            {
                bool parseOK;
                rData.push_back(cb.getAs<T>(parseOK, trim));
                if (!parseOK)
                {
                    return false;
                }
            }
            else
            {
                return false;
            }
        }
        return true;
    }
    return false;
}

template <typename T>
bool IndexingCSVParser::getIndexedRowAs(const size_t row, std::vector<T> &rData, TrimSpaceOption trim)
{
    return IndexingCSVParser::getIndexedRowColumnRangeAs<T>(row,0,numCols(row),rData, trim);
}

template <typename T>
bool IndexingCSVParser::getIndexedRowColumnRangeAs(const size_t row, const size_t startCol, const size_t numCols, std::vector<T> &rData, TrimSpaceOption trim)
{
    if (row < mSeparatorMatrix.size())
    {
        // Reserve data (will only increase reserved memory if needed, not shrink)
        rData.reserve(numCols);

        // Begin position
        size_t b = mSeparatorMatrix[row][startCol] + 1*(startCol > 0);
        // Move file pointer
        fseek(mpFile, b, SEEK_SET);
        // Character buffer for extraction and parsing
        CharBuffer cb;
        // Loop through each column on row
        for (size_t c=startCol+1; c<=startCol+numCols; ++c)
        {
            const size_t e = mSeparatorMatrix[row][c];
            cb.setContentSize(e-b);
            char* rc = fgets(cb.buff(), e-b+1, mpFile);
            if (rc)
            {
                bool parseOK;
                rData.push_back(cb.getAs<T>(parseOK, trim));
                if (!parseOK)
                {
                    return false;
                }
            }
            else
            {
                return false;
            }

            // Update b for next field, skipping the character itself
            b = mSeparatorMatrix[row][c]+1;
            // Move the file pointer, 1 char (gobble the separator)
            fgetc(mpFile);
        }
        return true;
    }
    return false;
}


//! @brief Extract the data of a given indexed position row and column (as given template argument)
//! @tparam T The type do convert asci text to
//! @param[in] row The row index (0-based)
//! @param[in] col The column index (0-based)
//! @param[in] trim Whether to trim leading and trailing spaces from data
//! @returns The value at the requested position as given template type argument default constructed value if position does not exist
template <typename T>
T IndexingCSVParser::getIndexedPosAs(const size_t row, const size_t col, bool &rParseOK, TrimSpaceOption trim)
{
    if (row < mSeparatorMatrix.size())
    {
        if (col+1 < mSeparatorMatrix[row].size())
        {
            // Begin and end positions
            size_t b = mSeparatorMatrix[row][col] + size_t(col > 0);
            size_t e = mSeparatorMatrix[row][col+1];
            fseek(mpFile, b, SEEK_SET);

            CharBuffer cb(e-b);
            char* rc = fgets(cb.buff(), e-b+1, mpFile);
            if (rc)
            {
                return cb.getAs<T>(rParseOK, trim);
            }
        }
    }
    rParseOK = false;
    return T();
}

template <typename T>
bool IndexingCSVParser::getRowAs(std::vector<T> &rData, TrimSpaceOption trim)
{
    bool isSuccess = true;
    CharBuffer cb;

    size_t b = ftell(mpFile);
    while (true)
    {
        size_t e = ftell(mpFile);
        int c = fgetc(mpFile);

        if (c == mSeparatorChar || c == '\n' || c == '\r' || c == EOF)
        {
            // Rewind file pointer to start of field
            fseek(mpFile, b, SEEK_SET);
            cb.setContentSize(e-b);
            char* rc = fgets(cb.buff(), e-b+1, mpFile);
            if (rc)
            {
                bool parseOK;
                rData.push_back(cb.getAs<T>(parseOK, trim));
                // Indicate we failed to parse, but we still need to gobble the entire line in case we reach EOF
                if (!parseOK)
                {
                    isSuccess = false;
                }
            }
            else
            {
                // Indicate we failed to parse, but we still need to gobble the entire line in case we reach EOF
                isSuccess = false;
            }

            // Eat the separator char, in case of CRLF EOL, then gobble both CR and expected LF
            do
            {
                c = fgetc(mpFile);
                b = ftell(mpFile); //!< @todo maybe can use +1 since binary mode (calc bytes) might be faster
            }while(c == '\r');

            // Break loop when we have reached EOL or EOF
            if (c == '\n' || c == EOF)
            {
                // If we got a LF then peek to see if EOF reached, if so gobble char to set EOF flag on file
                if (peek(mpFile) == EOF)
                {
                    fgetc(mpFile);
                }
                break;
            }
        }
    }

    //! @todo try to index line first before extracting data, might be faster since we can reserve (maybe)
    return isSuccess;
}

}

#endif // INDEXINGCSVPARSER_H
