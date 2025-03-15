#include "numhop/VariableStorage.h"
#include "numhop/Helpfunctions.h"

namespace numhop {

//! @brief Default constructor
VariableStorage::VariableStorage()
{
    mpExternalStorage = 0;
    mpParentStorage = 0;
}

//! @brief Reserve a value name, making it constant and impossible to change
//! @param[in] name The name of the value
//! @param[in] value The constant value
//! @returns True if the name could be reserved, false if it was already reserved
bool VariableStorage::reserveNamedValue(const std::string &name, double value)
{
    std::map<std::string,double>::iterator it = mReservedNameVauleMap.find(name);
    if (it == mReservedNameVauleMap.end())
    {
        mReservedNameVauleMap.insert(std::pair<std::string,double>(name, value));
        return true;
    }
    return false;
}

//! @brief Set a variable value
//! @param[in] name The name of the variable
//! @param[in] value The value
//! @param[out] rDidSetExternally Indicates if the variable was an external variable
//! @returns True if the variable was set, false otherwise
bool VariableStorage::setVariable(const std::string &name, double value, bool &rDidSetExternally)
{
    rDidSetExternally = false;

    // Check if name is reserved
    std::map<std::string,double>::iterator it = mReservedNameVauleMap.find(name);
    if (it != mReservedNameVauleMap.end())
    {
        return false;
    }

    // If not reserved, first try to set it externally
    if (mpExternalStorage)
    {
        rDidSetExternally = mpExternalStorage->setExternalValue(name, value);
    }

    // If we could not set externally, then set it internally
    if (!rDidSetExternally && isNameInternalValid(name))
    {
        std::map<std::string,double>::iterator it = mVariableMap.find(name);
        if (it == mVariableMap.end())
        {
            mVariableMap.insert(std::pair<std::string,double>(name, value));
            return true;
        }
        else
        {
            it->second = value;
            return true;
        }
    }
    return rDidSetExternally;
}

//! @brief Check if a given name is a valid internal storage name, based on given disallowed characters
//! @param[in] name The name to check
//! @returns True if the name is valid, else false
bool VariableStorage::isNameInternalValid(const std::string &name) const
{
    return !containsAnyof(name, mDisallowedInternalNameChars);
}

//! @brief Set disallowed characters in internal names
//! @param[in] disallowed A string containing disallowed characters
void VariableStorage::setDisallowedInternalNameCharacters(const std::string &disallowed)
{
    mDisallowedInternalNameChars = disallowed;
}

//! @brief Get the value of a variable or reserved constant value
//! @param[in] name The name of the variable
//! @param[out] rFound Indicates if the variable was found
//! @returns The value of the variable (if it was found, else a dummy value)
double VariableStorage::value(const std::string &name, bool &rFound) const
{
    rFound=false;

    // First try to find reserved variable internally
    std::map<std::string,double>::const_iterator it = mReservedNameVauleMap.find(name);
    if (it != mReservedNameVauleMap.end())
    {
        rFound = true;
        return it->second;
    }

    // Then try to find ordinary variable internally
    it = mVariableMap.find(name);
    if (it != mVariableMap.end())
    {
        rFound=true;
        return it->second;
    }

    // Else try to find it externally
    if (mpExternalStorage)
    {
        double value = mpExternalStorage->externalValue(name, rFound);
        if (rFound)
        {
            return value;
        }
    }

    return 0;
}

//! @brief Check if a given name is an existing variable (not reserved value)
//! @param[in] name The variable name to look for
//! @return true if found else false
bool VariableStorage::hasVariableName(const std::string &name) const
{
    // Try to find ordinary variable internally
    if (mVariableMap.find(name) != mVariableMap.end()) {
        return true;
    }

    // Else try to find it externally
    bool found = false;
    if (mpExternalStorage) {
        mpExternalStorage->externalValue(name, found);
    }
    return found;
}

//! @brief Set the external storage
//! @param[in] pExternalStorage A pointer to the external storage to use in variable lookup
void VariableStorage::setExternalStorage(ExternalVariableStorage *pExternalStorage)
{
    mpExternalStorage = pExternalStorage;
}

//! @brief Set the parent storage (not used yet)
//! @param[in] pParentStorage A pointer to the parent storage to use in variable lookup
//! @warning Not yet implemented
void VariableStorage::setParentStorage(VariableStorage *pParentStorage)
{
    mpParentStorage = pParentStorage;
}

//! @brief Clear the internal variable storage
void VariableStorage::clearInternalVariables()
{
    mVariableMap.clear();
}

ExternalVariableStorage::~ExternalVariableStorage() {

}

}
