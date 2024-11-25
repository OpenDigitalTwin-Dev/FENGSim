#include <iostream>

#include <IDTF/Converter.h>

int main()
{
    int resCode;
    std::string input = INPUT_FILE;
    std::string output = OUTPUT_FILE;
    bool res = IDTFConverter::IDTFToU3d(input, output, resCode, "../");
    return !res;
}
