#ifndef HELPFUNCTIONS_H
#define HELPFUNCTIONS_H

#include <string>
#include <list>

namespace numhop {

void extractExpressionRows(const std::string &script, const char &commentChar, std::list<std::string> &rScriptExpressions);
void removeAllWhitespaces(std::string &rString);
void stripLeadingTrailingWhitespaces(std::string &rString);
bool stripLeadingTrailingParanthesis(std::string &rString, bool &rDidStrip);
char stripInitialSign(std::string &rString);
void stripInitialPlus(std::string &rString);

template <typename ContainerT, typename KeyT>
bool contains(const ContainerT& container, const KeyT& key)
{
    return container.find(key) != container.end();
}

template <>
inline bool contains<std::string, char>(const std::string& container, const char& key)
{
    return container.find(key) != std::string::npos;
}

inline bool containsAnyof(const std::string &str, const std::string &match)
{
    return (str.find_first_of(match) != std::string::npos);
}

}

#endif // HELPFUNCTIONS_H
