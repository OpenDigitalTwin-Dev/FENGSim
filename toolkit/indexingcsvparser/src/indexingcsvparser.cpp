//!
//! @file   indexingcsvparser.cpp
//! @author Peter Nordin
//! @date   2015-02-03
//!
//! @brief Contains some of the IndexingCSVParser implementation
//!

#include "indexingcsvparser/indexingcsvparser.h"

#include <algorithm>
#include <string>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <limits>

using namespace std;
using namespace indcsvp;

namespace {

//! @brief Help function that gobbles all characters on a line
//! @param[in] pFile The file object to gobble from
void discardLine(FILE *pFile)
{
    while (fgetc(pFile) != '\n')
    {
        // Run loop till newline has been gobbled
    }
}


//! @brief Remove leading white spaces from a string
//! @param[in,out] rString The string to be modified
void trimLeadingSpaces(std::string& rString) {
    size_t count=0;
    for (size_t i=0; i<rString.size(); ++i) {
        if (isspace(rString[i])) {
            ++count;
        }
        else {
            break;
        }
    }
    if (count > 0) {
        rString.erase(0,count);
    }
}

} // End anon namespace

IndexingCSVParser::IndexingCSVParser()
{
    mpFile = 0;
    mSeparatorChar = ',';
    mNumSkipLines = 0;
    mCommentChar = '\0';
}

void IndexingCSVParser::setHeaderInfo(Direction direction, size_t rowOrColumn)
{
    mHeaderSetting.setHeaderInfo(direction, rowOrColumn);
}

//! @brief Set the separator character
//! @param[in] sep The separator character
void IndexingCSVParser::setSeparatorChar(char sep)
{
    mSeparatorChar = sep;
}

//! @brief Set the comment character
//! @param[in] sep The character to indicate a comment
void IndexingCSVParser::setCommentChar(char com)
{
    mCommentChar = com;
}

//! @brief Set the number of initial lines to ignore
//! @param[in] num The number of lines to ignore
void IndexingCSVParser::setNumLinesToSkip(size_t num)
{
    mNumSkipLines = num;
}

//! @brief Automatically choose the separator character
//! @param[in] rAlternatives A vector with alternatives, the first on encountered will be used.
//! @returns The chosen character, or the previously selected on if no match found
char IndexingCSVParser::autoSetSeparatorChar(const std::vector<char> &rAlternatives)
{
    // Discard header and comments
    readUntilData();

    bool found=false;
    while (!found && !feof(mpFile))
    {
        int c = fgetc(mpFile);
        for (size_t i=0; i<rAlternatives.size(); ++i)
        {
            if (c == rAlternatives[i])
            {
                mSeparatorChar = char(c);
                found = true;
                break;
            }
        }
    }
    return mSeparatorChar;
}

HeaderSetting IndexingCSVParser::getHeaderSetting() const
{
    return mHeaderSetting;
}

//! @brief Returns the separator character used
char IndexingCSVParser::getSeparatorChar() const
{
    return mSeparatorChar;
}

//! @brief Returns the comment character used
char IndexingCSVParser::getCommentChar() const
{
    return mCommentChar;
}

//! @brief Returns the number of initial lines to skip
size_t IndexingCSVParser::getNumLinesToSkip() const
{
    return mNumSkipLines;
}

//! @brief Open a file in binary read-only mode
//! @param[in] filePath Path to the file to open
//! @returns true if the files was opened successfully else false
bool IndexingCSVParser::openFile(const char *filePath)
{
    if (mpFile != 0) {
        closeFile();
    }
    mpFile = fopen(filePath, "rb");
    return (mpFile != 0);
}

void IndexingCSVParser::takeOwnershipOfFile(FILE* pExternalFile)
{
    if (mpFile != 0) {
        closeFile();
    }
    mpFile = pExternalFile;
}

//! @brief Close the opened file
void IndexingCSVParser::closeFile()
{
    if (mpFile)
    {
        fclose(mpFile);
        mpFile = 0;
    }
    mSeparatorMatrix.clear();
}

//! @brief Rewind the file pointer to the beginning of the file
void IndexingCSVParser::rewindFile()
{
    rewind(mpFile);
}

//! @brief Run indexing on the file, to find all separator positions
void IndexingCSVParser::indexFile()
{
    rewindFile();
    readUntilData();
    mSeparatorMatrix.clear();

    mSeparatorMatrix.reserve(100); //!< @todo guess num rows
    size_t lastLineNumSeparators = 20;

    // We register the position in the file before we read the char, as that will advance the file pointer
    size_t pos = ftell(mpFile);
    int c = fgetc(mpFile);
    while (c!=EOF)
    {
        // Append new line, but we will work with a reference
        mSeparatorMatrix.push_back(vector<size_t>());
        vector<size_t> &rLine = mSeparatorMatrix.back();
        rLine.reserve(lastLineNumSeparators);

        // Register Start of line position
        rLine.push_back(pos);
        // Now read line until and register each separator char position
        // If separator char == "space" then use special case
        if (mSeparatorChar == ' ')
        {
            bool lastWasSpace = true;
            while (c!='\n' && c!='\r' && c!=EOF)
            {
                if (c==mSeparatorChar)
                {
                    if (!lastWasSpace)
                    {
                        rLine.push_back(pos);
                    }
                    lastWasSpace = true;
                }
                else
                {
                    lastWasSpace = false;
                }

                // Get next char
                pos = ftell(mpFile);
                c = fgetc(mpFile);
            }
        }
        // else use this case for all ordinary separators
        else
        {
            while (c!='\n' && c!='\r' && c!=EOF)
            {
                if (c==mSeparatorChar)
                {
                    rLine.push_back(pos);
                }

                // Get next char
                pos = ftell(mpFile);
                c = fgetc(mpFile);
            }
        }

        // Register end of line position
        rLine.push_back(pos);
        // Read pos and first char on next line
        // The while loop make sure we gobble LF if we have CRLF eol
        while ( c == '\r' || c == '\n' )
        {
            pos = ftell(mpFile);
            c = fgetc(mpFile);
        }

        // Remember the length of the line (to reserve relevant amount of memory next time)
        lastLineNumSeparators =  rLine.size();
    }
}

//! @brief Returns the number of indexed data rows in the file
//! @note This will only work if the file has been indexed
//! @returns The number of indexed rows
size_t IndexingCSVParser::numRows() const
{
    return mSeparatorMatrix.size();
}

//! @brief Returns the number of indexed columns for a particular row
//! @note This will only work if the file has been indexed
//! @param[in] row The row index (0-based)
//! @returns The number of indexed columns on the requested row
size_t IndexingCSVParser::numCols(size_t row) const
{
    if (row < mSeparatorMatrix.size())
    {
        return mSeparatorMatrix[row].size()-1;
    }
    return 0;
}

//! @brief Check if all indexed rows have the same number of columns
//! @returns true if all rows have the same number of columns, else returns false
bool IndexingCSVParser::allRowsHaveSameNumCols() const
{
    const size_t nCols = numCols(0);
    for (size_t r=0; r<mSeparatorMatrix.size(); ++r)
    {
        if (numCols(r) != nCols)
        {
            return false;
        }
    }
    return true;
}


void IndexingCSVParser::minMaxNumCols(size_t &rMin, size_t &rMax)
{
    rMin = std::numeric_limits<size_t>::max();
    rMax = 0;
    for (size_t r=0; r<mSeparatorMatrix.size(); ++r)
    {
        rMin = std::min(rMin, mSeparatorMatrix[r].size());
        rMax = std::max(rMax, mSeparatorMatrix[r].size());
    }
    // Do not count end of line separator
    rMin -= 1;
    rMax -= 1;
}

const std::vector<string> &IndexingCSVParser::header() const
{
    return mHeader;
}

//! @brief Extract the data of a given indexed column (as std::string)
//! @param[in] col The column index (0-based)
//! @param[in,out] rData The data vector to append column data to
//! @param[in] trim Whether to trim leading and trailing spaces from data
//! @returns true if no errors occurred, else false
bool IndexingCSVParser::getIndexedColumn(const size_t col, std::vector<string> &rData, TrimSpaceOption trim)
{
    if (col < numCols(0))
    {
        const size_t nr = numRows();
        // Reserve data (will only increase reserved memmory if needed, not shrink)
        rData.reserve(nr);

        CharBuffer cb;

        // Loop through each row
        for (size_t r=0; r<nr; ++r)
        {
            // Begin and end positions
            size_t b = mSeparatorMatrix[r][col] + size_t(col > 0);
            size_t e = mSeparatorMatrix[r][col+1];
            // Move file ptr
            fseek(mpFile, b, SEEK_SET);

            // Extract data
            cb.setContentSize(e-b);
            char* rc = fgets(cb.buff(), e-b+1, mpFile);
            // Push back data
            if (rc)
            {
                rData.push_back(cb.str(trim));
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

//! @brief Extract the data of a given indexed row (as std::string)
//! @param[in] row The row index (0-based)
//! @param[in,out] rData The data vector to append row data to
//! @param[in] trim Whether to trim leading and trailing spaces from data
//! @returns true if no errors occurred, else false
bool IndexingCSVParser::getIndexedRow(const size_t row, std::vector<string> &rData, TrimSpaceOption trim)
{
    if (row < mSeparatorMatrix.size())
    {
        const size_t nc = numCols(row);
        // Reserve data (will only increase reserved memory if needed, not shrink)
        rData.reserve(nc);

        // Begin position
        size_t b = mSeparatorMatrix[row][0];
        // Move file ptr
        fseek(mpFile, b, SEEK_SET);
        // Character buffer for extraction
        CharBuffer cb;

        // Loop through each column on row
        for (size_t c=1; c<=nc; ++c)
        {
            const size_t e = mSeparatorMatrix[row][c];
            cb.setContentSize(e-b);
            char* rc = fgets(cb.buff(), e-b+1, mpFile);
            if (rc)
            {
                rData.push_back(cb.str(trim));
            }
            else
            {
                return false;
            }

            // Update b for next field, skipping the character itself
            b = mSeparatorMatrix[row][c]+1;
            // Move the file ptr, 1 char (gobble the separator)
            fgetc(mpFile);
        }
        return true;
    }
    return false;
}

//! @brief Extract the data of a given indexed position row and column (as std::string)
//! @param[in] row The row index (0-based)
//! @param[in] col The column index (0-based)
//! @param[in] trim Whether to trim leading and trailing spaces from data
//! @returns The value at the requested position as std::string or empty if position does not exist
string IndexingCSVParser::getIndexedPos(const size_t row, const size_t col, bool &rParseOK, TrimSpaceOption trim)
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
                rParseOK = true;
                return cb.str(trim);
            }
            else
            {
                rParseOK = false;
            }
        }
    }
    return "";
}

//! @brief Extract a data row from a non-indexed file (as std::string)
//! @param[in,out] rData The data vector to append extracted data to
//! @param[in] trim Whether to trim leading and trailing spaces from data
bool IndexingCSVParser::getRow(std::vector<string> &rData, TrimSpaceOption trim)
{
    return getRowAs(rData, trim);
}

//! @brief Check if more data rows are available for extraction (for non-indexed files)
//! @returns true if more rows are waiting, returns false if file pointer has reached EOF
bool IndexingCSVParser::hasMoreDataRows()
{
    return feof(mpFile) == 0;
}

bool IndexingCSVParser::getHeaderRow(std::vector<string> &rData)
{
    bool parseOK = getRow(rData, TrimLeadingTrailingSpace);
    if (parseOK) {
        // Strip comment char from first  header item if it is the first char
        if (mCommentChar != '\0' && !rData.empty() && !rData.front().empty()) {
            if (*rData.front().begin() == mCommentChar) {
                rData.front().erase(0,1);
                trimLeadingSpaces(rData.front());
            }
        }
    }
    return parseOK;
}

//! @brief Gobble the initial number of lines to skip and lines beginning with the comment character
void IndexingCSVParser::readUntilData()
{
    // First remove configured lines to skip
    skipNumLines(mNumSkipLines);

    const bool haveHeaderRow = (mHeaderSetting.isValid() && (mHeaderSetting.direction() == Row));
    size_t curentRowIndex = mNumSkipLines;

    // Now auto remove initial lines beginning with the comment char, but extract the header if it is among them
    if (mCommentChar != '\0') {
        size_t headerLine =  haveHeaderRow ? mHeaderSetting.rowOrColumn() : std::numeric_limits<size_t>::max();
        while (peek(mpFile) == mCommentChar) {
            if ( curentRowIndex == headerLine ) {
                getHeaderRow(mHeader);
            }
            else {
                discardLine(mpFile);
            }
            ++curentRowIndex;
        }
    }

    // Skip everything until the header line unless we have already found it
    if ( haveHeaderRow && (curentRowIndex <= mHeaderSetting.rowOrColumn()) ) {
        skipNumLines(mHeaderSetting.rowOrColumn()-curentRowIndex);
        getHeaderRow(mHeader);
        // There should be no non-data rows after the header in this case
    }
}

void IndexingCSVParser::skipNumLines(size_t num)
{
    for(size_t i=0; i<num; ++i) {
        discardLine(mpFile);
    }
}

HeaderSetting::HeaderSetting() {
    // Abusing size_t::max to indicate invalid, it is unlikely that a header will be placed at that line
    mHeaderIndex = std::numeric_limits<size_t>::max();
}

void HeaderSetting::setHeaderInfo(Direction direction, size_t rowOrCol) {
    mDirection = direction;
    mHeaderIndex = rowOrCol;
}

bool HeaderSetting::isValid() const {
    return mHeaderIndex != std::numeric_limits<size_t>::max();
}

Direction HeaderSetting::direction() const
{
    return mDirection;
}

size_t HeaderSetting::rowOrColumn() const
{
    return mHeaderIndex;
}
