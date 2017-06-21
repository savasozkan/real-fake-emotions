
#pragma once

#include <exception>
#include <string>
using namespace std;

class error : public exception
{
public:

	explicit error(const string& strMessage) throw() ;
	explicit error(const char* szMessage) throw() ;

	virtual ~error() throw() ;

	virtual const char* what() const throw() ;
	virtual const string Backtrace() const throw() ;

protected:

	void Initialize(const string& strMessage) throw() ;
	string m_strMessage ;
	static const int m_nMaximumBacktraceCount = 10 ;
	void *m_pBacktrace[m_nMaximumBacktraceCount + 2] ;
	int m_nBacktraceCount ;
};
