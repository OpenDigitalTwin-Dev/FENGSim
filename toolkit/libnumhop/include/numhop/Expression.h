#ifndef EXPRESSION_H
#define EXPRESSION_H

#include <string>
#include <list>
#include <set>
#include <vector>
#include "VariableStorage.h"

namespace numhop {

enum ExpressionOperatorT {AssignmentT, AdditionT, SubtractionT, MultiplicationT,
                          DivisionT, PowerT, LessThenT, GreaterThenT, OrT, AndT,
                          ValueT, FunctionCallT, UndefinedT};

class Expression
{
public:
    Expression();
    Expression(const Expression &other);
    Expression(const std::string &exprString, ExpressionOperatorT op);
    Expression(const std::string &leftExprString, const std::string &rightExprString, ExpressionOperatorT op);

    Expression& operator= (const Expression &other);

    bool empty() const;
    bool isValue() const;
    bool isNumericConstant() const;
    bool isNamedValue() const;
    bool isValid() const;

    const std::string &exprString() const;
    const std::string &leftExprString() const;
    const std::string &rightExprString() const;
    ExpressionOperatorT operatorType() const;

    double evaluate(VariableStorage &rVariableStorage, bool &rEvalOK) const;
    void extractNamedValues(std::set<std::string> &rNamedValues) const;
    void extractValidVariableNames(const VariableStorage &variableStorage, std::set<std::string> &rVariableNames) const;
    void replaceNamedValue(const std::string& oldName, const std::string& newName);

    std::string print();

protected:
    void commonConstructorCode();
    void copyFromOther(const Expression &other);

    std::string mLeftExpressionString, mRightExpressionString;
    std::list<Expression> mLeftChildExpressions, mRightChildExpressions;
    bool mHadLeftOuterParanthesis, mHadRightOuterParanthesis;
    bool mIsNumericConstant, mIsNamedValue, mIsValid;
    int mFunctionId;
    double mNumericConstantValue;
    ExpressionOperatorT mOperator;
};

bool interpretExpressionStringRecursive(std::string exprString, std::list<Expression> &rExprList);
bool interpretExpressionStringRecursive(std::string exprString, Expression &rExpr);
std::vector<std::string> getRegisteredFunctionNames();

}

#endif // EXPRESSION_H
