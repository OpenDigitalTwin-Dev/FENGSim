#include <iostream>
#include <string>
#include <list>
#include <vector>
#include <map>
#include <sstream>

#include "numhop.h"

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

template <typename ContainerT>
std::string printContainer(const ContainerT& values) {
    std::stringstream ss;
    typename ContainerT::const_iterator it;
    for (it=values.begin(); it!=values.end(); ++it) {
        ss << " " << *it;
    }
    return ss.str();
}

template <typename ValueContainerT, typename ExpectedContainerT>
void ensureContainsExpected(const ValueContainerT& values, const ExpectedContainerT& expectedValues) {
    if (values.size() != expectedValues.size()) {
        std::stringstream ss;
        ss << "Container size missmatch: " << values.size() << " != " << expectedValues.size() << std::endl;
        ss << "Actual  : " << printContainer(values) << std::endl;
        ss << "Expected: " << printContainer(expectedValues) << std::endl;
        FAIL(ss.str().c_str());

    }
    typename ExpectedContainerT::const_iterator it;
    for (it=expectedValues.begin(); it!=expectedValues.end(); ++it) {
        bool foundExpected = values.find(*it)!=values.end();
        if (!foundExpected) {
            std::stringstream ss;
            ss << "Did not find expected value: ";
            ss << *it;
            ss << " among: " << printContainer(values);
            FAIL(ss.str().c_str());
        }
    }
}



// Local variable storage wrapper class
// local means local variables in this application,
// they are seen as external variables inside libnumhop
class ApplicationVariables : public numhop::ExternalVariableStorage
{
public:
  // Overloaded -----
  double externalValue(std::string name, bool &rFound) const
  {
    std::map<std::string, double>::const_iterator it = mVars.find(name);
    if (it != mVars.end()) {
      rFound = true;
      return mVars.at(name);
    }
    rFound = false;
    return 0;
  }

  bool setExternalValue(std::string name, double value)
  {
    std::map<std::string, double>::iterator it = mVars.find(name);
    if (it != mVars.end()) {
      it->second = value;
      return true;
    }
    return false;
  }
  // -----


  void addVariable(std::string name, double value)
  {
      mVars.insert(std::pair<std::string,double>(name, value));
  }

  double& operator[](const std::string& name)
  {
      return mVars[name];
  }

private:
  std::map<std::string, double> mVars;
};

void test_allok(const std::string &exprs, const double expected_result, numhop::VariableStorage &variableStorage){

  std::list<std::string> exprlist;
  numhop::extractExpressionRows(exprs, '#', exprlist);

  INFO("Full expression: " << exprs);
  double value_first_time, value_second_time;
  std::list<std::string>::iterator it;
  for (it=exprlist.begin(); it!=exprlist.end(); ++it) {
    INFO("Current sub expression: " << *it);
    numhop::Expression e;
    bool interpretOK = numhop::interpretExpressionStringRecursive(*it, e);
    REQUIRE(interpretOK == true);

    bool first_eval_ok, second_eval_ok;;
    value_first_time = e.evaluate(variableStorage, first_eval_ok);
    REQUIRE(first_eval_ok == true);
    // evaluate again, should give same result
    value_second_time = e.evaluate(variableStorage, second_eval_ok);
    REQUIRE(second_eval_ok == true);

    REQUIRE(value_first_time == Approx(value_second_time));

    // Test expression.print
    //WARN(e.print());
  }

  REQUIRE(value_first_time == Approx(expected_result));
}

void test_interpret_fail(const std::string &expr)
{
  INFO("Full expression: " << expr);
  numhop::Expression e;
  bool interpretOK = numhop::interpretExpressionStringRecursive(expr, e);
  REQUIRE(interpretOK == false);
  //todo what happens if e is evaluated ?
}

void test_eval_fail(const std::string &expr, numhop::VariableStorage &variableStorage)
{
  INFO("Full expression: " << expr);
  numhop::Expression e;
  bool interpretOK = numhop::interpretExpressionStringRecursive(expr, e);
  REQUIRE(interpretOK == true);
  bool first_eval_ok;
  double value_first_time = e.evaluate(variableStorage, first_eval_ok);
  REQUIRE(first_eval_ok == false);
}

void test_extract_variablenames(const std::string &expr, numhop::VariableStorage &variableStorage, std::list<std::string> expectednames, std::list<std::string> expectedvalidvars)
{
    std::set<std::string> valuenames, validvarnames;
    numhop::Expression e;
    bool interpretOK = numhop::interpretExpressionStringRecursive(expr, e);
    REQUIRE(interpretOK == true);
    e.extractNamedValues(valuenames);
    ensureContainsExpected(valuenames, expectednames);

    e.extractValidVariableNames(variableStorage, validvarnames);
    ensureContainsExpected(validvarnames, expectedvalidvars);
}

void printRegisteredFunctionNames()
{
    std::string allnames;
    std::vector<std::string> names = numhop::getRegisteredFunctionNames();
    for(size_t i=0; i<names.size(); ++i) {
        allnames.append(names[i]+' ');
    }
    WARN(allnames.c_str());
}


TEST_CASE("Variable Assignment") {
  numhop::VariableStorage vs;
  test_allok("a=5;a=8;a;", 8, vs);
  test_allok("a=6;\n a=7.14\n a;", 7.14, vs);
}

TEST_CASE("External Variables") {
  numhop::VariableStorage vs;

  ApplicationVariables av;
  av.addVariable("dog", 55);
  av.addVariable("cat", 66);
  vs.setExternalStorage(&av);

  test_allok("dog", 55, vs);
  test_allok("cat", 66, vs);

  test_allok("dog=4; 1-(-2-3-(-dog-5.1))", -3.1, vs);
  test_allok("-dog", -4, vs);
  REQUIRE(av["dog"] == 4);

  test_allok("cat \n dog \r dog=5;cat=2.1;a=3;b=dog*cat*a;b", 31.5, vs);
  REQUIRE(av["cat"] == 2.1);
}

TEST_CASE("Reserved Variable") {
  numhop::VariableStorage vs;
  vs.reserveNamedValue("pi", 3.1415);
  ApplicationVariables av;
  vs.setExternalStorage(&av);

  // Add external pi variable
  av.addVariable("pi", 10000);

  // Here the reserved pi should be used, not the external one
  test_allok("pi", 3.1415, vs);
  test_allok("a=pi*2", 6.283, vs);

  // It should not be possible to change the external pi, or the reserved value
  test_eval_fail("pi=123", vs);
  test_allok("pi", 3.1415, vs);
  REQUIRE(av["pi"] == 10000 );
}

TEST_CASE("Expression Parsing") {
  numhop::VariableStorage vs;
  std::string expr = " \t #   \n    a=5;\n #   a=8\n a+1; \r\n a+2 \r a+3 \r\n #Some comment ";

  // Extract individual expression rows, treat # as comment
  // ; \r \n breaks lines
  // The expression above contains 4 actual valid sub expressions
  std::list<std::string> exprlist;
  numhop::extractExpressionRows(expr, '#', exprlist);
  REQUIRE(exprlist.size() == 4);

  // Result should be a+3 with a = 5
  test_allok(expr, 8, vs);
}

TEST_CASE("Various Math") {
  numhop::VariableStorage vs;

  test_allok("2^2", 4, vs);
  test_allok("2^(1+1)", 4, vs);
  test_allok("7/3/4/5", 0.11666667, vs);
  test_allok("7/(3/(4/5))", 1.8666667, vs);
  test_allok("(4/3*14*7/3/4/5*5/(4*3/2))", 1.8148148148148147, vs);
  test_allok("1-2*3-3*4-4*5;", -37, vs);
  test_allok("1-(-2-3-(-4-5))", -3, vs);
  test_allok("-1-2-3*4-4-3", -22, vs);
  test_allok("-1-(2-3)*4-4-3", -4, vs);
  test_allok("-(((-2-2)-3)*4)", 28, vs);

  // Test use of multiple + -
  test_allok("2--3;", 5, vs);
  test_allok("1+-3", -2, vs);
  test_allok("1-+3", -2, vs);
  test_allok("1++3", 4, vs);
  test_allok("1---3", -2, vs);

  test_allok("a=1;b=2;c=3;d=a+b*c;d",7, vs);
  // Reuse b and a from last expression (stored in vs)
  test_allok("b^b;", 4, vs);
  test_allok("b^(a+a)", 4, vs);
  test_allok("a-b;", -1, vs);
  test_allok("a-b+a", 0, vs);
  test_allok("-a+b", 1, vs);
  test_allok("b-a;", 1, vs);
  test_allok("(-a)+b", 1, vs);
  test_allok("b+(-a);", 1, vs);
  test_allok("b+(+a)", 3, vs);
  test_allok("b+a", 3, vs);
  test_allok("+a", 1, vs);
  test_allok("0-(a-b)+b", 3, vs);

  test_allok("a=1;b=2;c=3;d=4; a-b+c-d+a", -1, vs);
  test_allok("a=1;b=2;c=3;d=4; a-b-c-d+a", -7, vs);
  test_allok("a=1;b=2;c=3;d=4;a=(3+b)+4^2*c^(2+d)-7*(d-1)", 11648, vs);
  // Reuse resulting a from last expression 
  test_allok("b=2;c=3;d=4;a=(3+b)+4^2*c^(2+d)-7/6/5*(d-1)", 11668.299999999999, vs);

  test_allok("value=5;", 5, vs);
  test_allok("value+2;", 7, vs);
  test_allok("value-2", 3, vs);
  test_allok("value*1e+2", 500, vs);
}

TEST_CASE("Exponential Notation") {
  numhop::VariableStorage vs;
  test_allok("2e-2-1E-2", 0.01, vs);
  test_allok("1e+2+3E+2", 400, vs);
  test_allok("1e2+1E2", 200, vs);
}

TEST_CASE("Boolean Expressions") {
  numhop::VariableStorage vs;

  test_allok("2<3", 1, vs);
  test_allok("2<2", 0, vs);
  test_allok("4.2>2.5", 1, vs);

  // Note the difference on how the minus sign is interpreted
  test_allok("(-4.2)>3", 0, vs);
  // In this second case, the expression is treated as -(4.2>3)
  test_allok("-4.2>3", -1, vs);

  test_allok("1|0", 1, vs);
  test_allok("0|0", 0, vs);
  test_allok("0|0|0|1", 1, vs);
  test_allok("1|1|1|1|1", 1, vs);
  test_allok("2|3|0", 1, vs);
  test_allok("(-2)|3", 1, vs);
  test_allok("(-1)|(-2)", 0, vs);

  test_allok("1&0", 0, vs);
  test_allok("0&0", 0, vs);
  test_allok("(-1)&1.5", 0, vs);
  test_allok("1&1", 1, vs);
  test_allok("1&1&1&0.4", 0, vs);
  test_allok("1&0&1", 0, vs);

  test_allok("2<3 | 4<2", 1, vs);
  test_allok("2<3 & 4<2", 0, vs);
  test_allok("2<3 & 4>2", 1, vs);
  test_allok("x=2.5; (x>2&x<3)*1+(x>3&x<4)*2", 1, vs);
  test_allok("x=3.6; (x>2&x<3)*1+(x>3&x<4)*2", 2, vs);
}

TEST_CASE("Disallowed Characters") {
    numhop::VariableStorage vs;
    vs.setDisallowedInternalNameCharacters("cg");
    test_allok("ape=5;", 5, vs);
    test_eval_fail("cat=1", vs);
    test_eval_fail("dog=1", vs);
    test_allok("a.b=5;", 5, vs);
    vs.setDisallowedInternalNameCharacters(".");
    test_allok("cat=1", 1, vs);
    test_allok("dog=1", 1, vs);
    test_eval_fail("a.b=5;", vs);
}

TEST_CASE("Extract variable names") {

  numhop::VariableStorage vs;
  vs.reserveNamedValue("pi", 3.1415);
  bool setOK;
  vs.setVariable("ivar1", 1, setOK);
  vs.setVariable("ivar2", 2, setOK);
  ApplicationVariables av;
  av.addVariable("evar1", 1);
  av.addVariable("evar2", 2);
  vs.setExternalStorage(&av);

  std::list<std::string> expectedNamedValues, expectedValidVariableNames;
  expectedNamedValues.push_back("ivar1");
  expectedValidVariableNames = expectedNamedValues;
  test_extract_variablenames("ivar1", vs, expectedNamedValues, expectedValidVariableNames);
  test_extract_variablenames("-ivar1", vs, expectedNamedValues, expectedValidVariableNames);

  expectedNamedValues.push_back("ivar2");
  expectedNamedValues.push_back("evar1");
  expectedNamedValues.push_back("evar2");
  expectedNamedValues.push_back("pi");
  expectedNamedValues.push_back("invalid");
  expectedValidVariableNames = expectedNamedValues;
  // remove pi and invalid
  expectedValidVariableNames.pop_back();
  expectedValidVariableNames.pop_back();
  const char* expr = "ivar2 = (ivar1 + evar1 + evar2 -3*ivar1)*pi^invalid";
  test_extract_variablenames(expr, vs, expectedNamedValues, expectedValidVariableNames);
}

TEST_CASE("Rename named values") {

  std::vector<std::string> expectedNamedValues;
  expectedNamedValues.push_back("ivar1");
  expectedNamedValues.push_back("ivar2");
  expectedNamedValues.push_back("evar1");
  expectedNamedValues.push_back("evar2");
  expectedNamedValues.push_back("pi");
  const char* expr = "ivar2 = (ivar1 + evar1 + 2 + evar2 -3*ivar1)*pi";

  numhop::Expression e;
  bool interpretOK = numhop::interpretExpressionStringRecursive(expr, e);
  REQUIRE(interpretOK == true);

  std::set<std::string> namedValues;
  e.extractNamedValues(namedValues);
  ensureContainsExpected(namedValues, expectedNamedValues);

  namedValues.clear();
  e.replaceNamedValue("ivar1", "replace1");
  e.extractNamedValues(namedValues);
  expectedNamedValues[0] = "replace1";
  ensureContainsExpected(namedValues, expectedNamedValues);

  namedValues.clear();
  e.replaceNamedValue("ivar2", "replace2");
  e.extractNamedValues(namedValues);
  expectedNamedValues[1] = "replace2";
  ensureContainsExpected(namedValues, expectedNamedValues);

  namedValues.clear();
  e.replaceNamedValue("pi", "replace3");
  e.extractNamedValues(namedValues);
  expectedNamedValues[4] = "replace3";
  ensureContainsExpected(namedValues, expectedNamedValues);
}

TEST_CASE("Math functions") {
  numhop::VariableStorage vs;
  vs.reserveNamedValue("pi", 3.1415);

  test_allok("cos(0)", 1, vs);
  test_allok("(cos((0)))", 1, vs);
  test_allok("cos(0)+cos(0)", 2, vs);
  test_allok("cos( 0 * 100)", 1, vs);
  test_allok("cos(sin(0))", 1, vs);
  test_allok("sin(pi/2*cos(0))", 1, vs);
  test_allok("atan2(0,1)", 0, vs);
  test_allok("atan2( (1+1)/2 , 2^0 )", 0.785398163397, vs);
  test_allok("floor(6.7)", 6, vs);
  test_allok("ceil(6.2)", 7, vs);
  test_allok("abs(-6.7)", 6.7, vs);
  test_allok("min(6,7)", 6, vs);
  test_allok("max(6,7)", 7, vs);

}

TEST_CASE("Expressions that should fail") {
  numhop::VariableStorage vs;

  test_interpret_fail("2*-2");
  test_interpret_fail("a += 5");
  test_interpret_fail("1+1-");
  test_interpret_fail(" = 5");
  test_interpret_fail("flooor(6.7)");
  test_interpret_fail("floor(6,7)");
  test_interpret_fail("atan2(1)");

  test_eval_fail("floor6.7)", vs);  //!< @todo should fail interpret
  test_eval_fail("floor(6.7", vs);  //!< @todo should fail interpret
  test_eval_fail(" cos((0+1) ", vs); //!< @todo should fail interpret
  test_eval_fail("0.5huj", vs);
}
