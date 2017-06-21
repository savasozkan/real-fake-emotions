
#pragma once

#include <string>
#include <vector>
using namespace std ;

class path
{
public:

	explicit path() ;
	path(const char* szPath) ;
	path(const string& strPath) ;
	explicit path(const path& pathDirectory, const path& pathFileName) ;
	~path() ;

	const path Root() const ;
	const path Directory() const ;
	const path BaseName() const ;
	const path RawName() const ;
	const path Extension() const ;
	const vector<path> Glob() const ;

	operator const char*() const ;
	const path operator +(const path& path) const ;
	void operator +=(const path& path) ;
	void operator +=(const string& strText) ;
	bool operator ==(const path& path) const ;
	bool operator ==(const char* szText) const ;
	bool operator !=(const path& path) const ;
	bool operator !=(const char* szText) const ;

	bool operator <(const path& path) const ;
	bool operator <(const char* szText) const ;
	bool operator <=(const path& path) const ;
	bool operator <=(const char* szText) const ;
	bool operator >(const path& path) const ;
	bool operator >(const char* szText) const ;
	bool operator >=(const path& path) const ;
	bool operator >=(const char* szText) const ;

	static const path ApplicationPath() ;
	static const path UserHomeDirectory() ;
	static const path TemporaryFile() ;

private:

	const string UnixSeparated(const string& strPath) const ;
	const string LastSeparationStrip(const string& strPath) const ;
	void GetStat(void* stat) const ;
	string m_strPath ;
} ;

const path operator +(const string& strText, const path& path) ;
