
#include <execinfo.h>
#include <stdlib.h>
#include "error.h"

error::error(const string& strMessage) throw()
{
	Initialize(strMessage) ;
}

error::error(const char* szMessage) throw()
{
	Initialize(string(szMessage)) ;
}

void error::Initialize(const string& strMessage) throw()
{
	m_strMessage = strMessage ;
	m_nBacktraceCount = backtrace(m_pBacktrace, m_nMaximumBacktraceCount + 2) ;
}

error::~error() throw()
{}

const char* error::what() const throw()
{
	return m_strMessage.c_str() ;
}

const string error::Backtrace() const throw()
{
	char **pSymbols = backtrace_symbols(m_pBacktrace, m_nBacktraceCount) ;
	string strBacktrace = "Backtrace:\n";
	if(pSymbols)
	{
		for(int i = 2 ; i < m_nBacktraceCount ; i += 1)
		{
			strBacktrace += string("    ") + pSymbols[i] + '\n' ;
		}
		free(pSymbols) ;
	}
	return strBacktrace ;
}
