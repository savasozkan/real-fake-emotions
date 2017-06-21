
#include <cstdlib>
#include <sys/stat.h>
#include <glob.h>
#include <errno.h>
#include <unistd.h>
#include <stdio.h>
#include <pwd.h>
#include "path.h"
#include "error.h"


path::path()
{}

path::path(const char* szPath)
{
	if(not szPath)
		throw error("[path] Invalid pointer.") ;

	m_strPath = UnixSeparated(szPath) ;
}

path::path(const string& strPath)
{
	m_strPath = UnixSeparated(strPath) ;
}

path::path(const path& pathDirectory, const path& pathFileName)
{
	m_strPath = UnixSeparated(string(LastSeparationStrip(string(pathDirectory)) + "/" + pathFileName)) ;
}

path::~path()
{}

const path path::Root() const
{
	size_t nFoundAt = m_strPath.find('/') + 1 ;
	while(nFoundAt < m_strPath.length() and m_strPath[nFoundAt] == '/')
		nFoundAt += 1 ;
	if(nFoundAt != string::npos)
		return path(m_strPath.substr(0, nFoundAt)) ;
	else
		return path() ;
}

const path path::Directory() const
{
	size_t nFoundAt = LastSeparationStrip(m_strPath).rfind('/') ;
	if(nFoundAt != string::npos)
		return path(LastSeparationStrip(m_strPath).substr(0, nFoundAt)) ;
	else
		return path(".") ;
}

const path path::BaseName() const
{
	size_t nFoundAt = LastSeparationStrip(m_strPath).rfind('/') ;
	if(nFoundAt != string::npos)
		return path(LastSeparationStrip(m_strPath).substr(nFoundAt + 1)) ;
	else
		return *this ;
}

const path path::RawName() const
{
	path base = BaseName() ;

	size_t nFoundAt = base.m_strPath.rfind('.') ;
	if(nFoundAt != string::npos)
		return path(base.m_strPath.substr(0, nFoundAt)) ;
	else
		return base ;
}

const path path::Extension() const
{
	size_t nFoundAt = LastSeparationStrip(m_strPath).rfind('.') ;
	if(nFoundAt != string::npos)
		return path(LastSeparationStrip(m_strPath).substr(nFoundAt + 1)) ;
	else
		return path() ;
}

const vector<path> path::Glob() const
{
	glob_t fileGlob ;
	int nReturnValue = glob(*this, 0, 0, &fileGlob) ;

	if(not (nReturnValue == 0 or nReturnValue == GLOB_NOMATCH))
		throw error("[path] Can't obtain glob: " + m_strPath) ;

	vector<path> glob(fileGlob.gl_pathv, fileGlob.gl_pathv + fileGlob.gl_pathc) ;
	globfree(&fileGlob) ;

	return glob ;
}

const string path::LastSeparationStrip(const string& strPath) const
{
	if(m_strPath.length() == 0 or m_strPath[m_strPath.length() - 1] != '/')
		return strPath ;
	else
		return strPath.substr(0, strPath.length() - 1) ;
}

const string path::UnixSeparated(const string& strPath) const
{
	bool bLastWasSeparation = false ;

	string strResult ;
	for(int i = 0 ; i < int(strPath.size()) ; i += 1)
	{
		if(strPath[i] == '/' or strPath[i] == '\\')
		{
			if(not bLastWasSeparation) strResult += '/' ;
			bLastWasSeparation = true ;
		}
		else
		{
			strResult += strPath[i] ;
			bLastWasSeparation = false ;
		}
	}

	return strResult ;
}

void path::GetStat(void* stat) const
{
	if(stat64(*this, reinterpret_cast<struct stat64*>(stat)) < 0)
		throw error("[path] Can't obtain stats for file: " + m_strPath) ;
}

path::operator const char*() const
{
	return m_strPath.c_str() ;
}

const path path::operator +(const path& pat) const
{
	return path(m_strPath + pat.m_strPath) ;
}

void path::operator +=(const path& path)
{
	m_strPath += path.m_strPath ;
}

void path::operator +=(const string& strText)
{
	m_strPath += UnixSeparated(strText) ;
}

const path operator +(const string& strText, const path& pat)
{
	return path(string(path(strText)) + string(pat)) ;
}

bool path::operator ==(const path& path) const
{
	return m_strPath == path.m_strPath ;
}

bool path::operator ==(const char* szText) const
{
	return m_strPath == szText ;
}

bool path::operator !=(const path& path) const
{
	return m_strPath != path.m_strPath ;
}

bool path::operator !=(const char* szText) const
{
	return m_strPath != szText ;
}

bool path::operator <(const path& path) const
{
	return m_strPath < path.m_strPath ;
}

bool path::operator <(const char* szText) const
{
	return m_strPath < szText ;
}

bool path::operator <=(const path& path) const
{
	return m_strPath <= path.m_strPath ;
}

bool path::operator <=(const char* szText) const
{
	return m_strPath <= szText ;
}
bool path::operator >(const path& path) const
{
	return m_strPath > path.m_strPath ;
}

bool path::operator >(const char* szText) const
{
	return m_strPath > szText ;
}

bool path::operator >=(const path& path) const
{
	return m_strPath >= path.m_strPath ;
}

bool path::operator >=(const char* szText) const
{
	return m_strPath >= szText ;
}

const path path::ApplicationPath()
{
	return path(program_invocation_name) ;
}

const path path::UserHomeDirectory()
{
	errno = 0 ;
	struct passwd* pwd = getpwuid(geteuid()) ;
	if(not pwd)
	{
		string strError ;
		switch(errno)
		{
		case EIO    : strError = "I/O error."                     ; break ;
		case EINTR  : strError = "Interrupted with signal."       ; break ;
		case EMFILE : strError = "All file descriptors are open." ; break ;
		case ENFILE : strError = "No more files can be opened."   ; break ;
		default     : strError = "Unknown error."                 ; break ;
		}
		throw error("[path] Can't obtain user home. " + strError) ;
	}
	return path(pwd->pw_dir) ;
}

const path path::TemporaryFile()
{
	path temp("/tmp/path_temp_XXXXXX") ;
	bool bDone = false ;

	{
		temp = path("/tmp/path_temp_XXXXXX") ;
		int nFile = -1 ;
		nFile = mkstemp(&temp.m_strPath[0]) ;
		bDone = nFile >= 0 ;
		close(nFile) ;
	}
	/// Try current folder.
	if(not bDone)
	{
		temp = path("./path_temp_XXXXXX") ;
		int nFile = -1 ;
		nFile = mkstemp(&temp.m_strPath[0]) ;
		bDone = nFile >= 0 ;
		close(nFile) ;
	}

	if(not bDone)
		throw error("[path] Can't find a temporary file name.") ;
	return temp ;
}
