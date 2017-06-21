
#pragma once

#include <vector>
#include <cmath>
#include <complex>
#include <string>
#include <iostream>
#include <algorithm>
#include <utility>
#include <memory>

#include "error.h"
using namespace std ;


template<typename T>
const vector<T> operator +(const vector<T>& v1, const vector<T>& v2)
{
	vector<T> vResult(v1) ;
	vResult += v2 ;
	return vResult ;
}

template<typename T, typename V>
const vector<T> operator +(const vector<T>& v, V value)
{
	vector<T> vResult(v) ;
	vResult += value ;
	return vResult ;
}

template<typename T>
const vector<T> operator +(T value, const vector<T>& v)
{
	vector<T> vResult(v.size()) ;
	for(int i = 0 ; i < int(vResult.size()) ; i += 1)
		vResult[i] = value + v[i] ;
	return vResult ;
}

template<typename T>
void operator +=(vector<T>& v1, const vector<T>& v2)
{
	if(v1.size() != v2.size())
		throw error("[operators] Vectors have different lengths.") ;

	for(int i = 0 ; i < int(v1.size()) ; i += 1)
		v1[i] += v2[i] ;
}

template<typename T, typename V>
void operator +=(vector<T>& v, V value)
{
	for(int i = 0 ; i < int(v.size()) ; i += 1)
		v[i] += value ;
}

template<typename T>
const vector<T> operator -(const vector<T>& v1, const vector<T>& v2)
{
	vector<T> vResult(v1) ;
	vResult -= v2 ;
	return vResult ;
}

template<typename T, typename V>
const vector<T> operator -(const vector<T>& v, V value)
{
	vector<T> vResult(v) ;
	vResult -= value ;
	return vResult ;
}

template<typename T>
const vector<T> operator -(T value, const vector<T>& v)
{
	vector<T> vResult(v.size()) ;
	for(int i = 0 ; i < int(vResult.size()) ; i += 1)
		vResult[i] = value - v[i] ;
	return vResult ;
}

template<typename T>
void operator -=(vector<T>& v1, const vector<T>& v2)
{
	if(v1.size() != v2.size())
		throw error("[operators] Vectors have different lengths.") ;

	for(int i = 0 ; i < int(v1.size()) ; i += 1)
		v1[i] -= v2[i] ;
}

template<typename T, typename V>
void operator -=(vector<T>& v, V value)
{
	for(int i = 0 ; i < int(v.size()) ; i += 1)
		v[i] -= value ;
}

template<typename T>
const vector<T> operator *(const vector<T>& v1, const vector<T>& v2)
{
	vector<T> vResult(v1) ;
	vResult *= v2 ;
	return vResult ;
}

template<typename T, typename V>
const vector<T> operator *(const vector<T>& v, V value)
{
	vector<T> vResult(v) ;
	vResult *= value ;
	return vResult ;
}

template<typename T>
const vector<T> operator *(T value, const vector<T>& v)
{
	vector<T> vResult(v.size()) ;
	for(int i = 0 ; i < int(vResult.size()) ; i += 1)
		vResult[i] = value * v[i] ;
	return vResult ;
}

template<typename T>
void operator *=(vector<T>& v1, const vector<T>& v2)
{
	if(v1.size() != v2.size())
		throw error("[operators] Vectors have different lengths.") ;

	for(int i = 0 ; i < int(v1.size()) ; i += 1)
		v1[i] *= v2[i] ;
}

template<typename T, typename V>
void operator *=(vector<T>& v, V value)
{
	for(int i = 0 ; i < int(v.size()) ; i += 1)
		v[i] *= value ;
}

template<typename T>
const vector<T> operator /(const vector<T>& v1, const vector<T>& v2)
{
	vector<T> vResult(v1) ;
	vResult /= v2 ;
	return vResult ;
}

template<typename T, typename V>
const vector<T> operator /(const vector<T>& v, V value)
{
	vector<T> vResult(v) ;
	vResult /= value ;
	return vResult ;
}

template<typename T>
const vector<T> operator /(T value, const vector<T>& v)
{
	vector<T> vResult(v.size()) ;
	for(int i = 0 ; i < int(vResult.size()) ; i += 1)
		vResult[i] = value / v[i] ;
	return vResult ;
}

template<typename T>
void operator /=(vector<T>& v1, const vector<T>& v2)
{
	if(v1.size() != v2.size())
		throw error("[operators] Vectors have different lengths.") ;

	for(int i = 0 ; i < int(v1.size()) ; i += 1)
		v1[i] /= v2[i] ;
}

template<typename T, typename V>
void operator /=(vector<T>& v, V value)
{
	for(int i = 0 ; i < int(v.size()) ; i += 1)
		v[i] /= value ;
}

template<typename T>
const vector<T> operator +(const vector<T>& v)
{
	return v ;
}

template<typename T>
const vector<T> operator -(const vector<T>& v)
{
	vector<T> vNegated(v.size()) ;
	for(int i = 0 ; i < int(vNegated.size()) ; i += 1)
		vNegated[i] = -v[i] ;
	return vNegated ;
}

template<typename T>
bool operator ==(const vector<T>& v1, const vector<T>& v2)
{
	if(v1.size() != v2.size())
		return false ;

	for(int i = 0 ; i < int(v1.size()) ; i += 1)
		if(v1[i] != v2[i]) return false ;
	return true ;
}

template<typename T>
bool operator !=(const vector<T>& v1, const vector<T>& v2)
{
	return !(v1 == v2) ;
}

template<typename T>
const vector<T> operator |(const vector<T>& v1, const vector<T>& v2)
{
	vector<T> vResult(v1) ;
	vResult |= v2 ;
	return vResult ;
}

template<typename T>
void operator |=(vector<T>& v1, const vector<T>& v2)
{
	v1.insert(v1.end(), v2.begin(), v2.end()) ;
}

template<typename T, typename U>
const pair<T, U> operator +(const pair<T, U>& p1, const pair<T, U>& p2)
{
	return pair<T, U>(p1.first + p2.first, p1.second + p2.second) ;
}

template<typename T, typename U, typename V>
const pair<T, U> operator +(const pair<T, U>& p, V value)
{
	return pair<T, U>(p.first + value, p.second + value) ;
}

template<typename T, typename U, typename V>
const pair<T, U> operator +(V value, const pair<T, U>& p)
{
	return pair<T, U>(value + p.first, value + p.second) ;
}

template<typename T, typename U>
void operator +=(pair<T, U>& p1, const pair<T, U>& p2)
{
	p1.first  += p2.first  ;
	p1.second += p2.second ;
}

template<typename T, typename U, typename V>
void operator +=(pair<T, U>& p, V value)
{
	p.first  += value ;
	p.second += value ;
}

template<typename T, typename U>
const pair<T, U> operator -(const pair<T, U>& p1, const pair<T, U>& p2)
{
	return pair<T, U>(p1.first - p2.first, p1.second - p2.second) ;
}

template<typename T, typename U, typename V>
const pair<T, U> operator -(const pair<T, U>& p, V value)
{
	return pair<T, U>(p.first - value, p.second - value) ;
}

template<typename T, typename U, typename V>
const pair<T, U> operator -(V value, const pair<T, U>& p)
{
	return pair<T, U>(value - p.first, value - p.second) ;
}

template<typename T, typename U>
void operator -=(pair<T, U>& p1, const pair<T, U>& p2)
{
	p1.first  -= p2.first  ;
	p1.second -= p2.second ;
}

template<typename T, typename U, typename V>
void operator -=(pair<T, U>& p, V value)
{
	p.first  -= value ;
	p.second -= value ;
}

template<typename T, typename U>
const pair<T, U> operator *(const pair<T, U>& p1, const pair<T, U>& p2)
{
	return pair<T, U>(p1.first * p2.first, p1.second * p2.second) ;
}

template<typename T, typename U, typename V>
const pair<T, U> operator *(const pair<T, U>& p, V value)
{
	return pair<T, U>(p.first * value, p.second * value) ;
}

template<typename T, typename U, typename V>
const pair<T, U> operator *(V value, const pair<T, U>& p)
{
	return pair<T, U>(value * p.first, value * p.second) ;
}

template<typename T, typename U>
void operator *=(pair<T, U>& p1, const pair<T, U>& p2)
{
	p1.first  *= p2.first  ;
	p1.second *= p2.second ;
}

template<typename T, typename U, typename V>
void operator *=(pair<T, U>& p, V value)
{
	p.first  *= value ;
	p.second *= value ;
}

template<typename T, typename U>
const pair<T, U> operator /(const pair<T, U>& p1, const pair<T, U>& p2)
{
	return pair<T, U>(p1.first / p2.first, p1.second / p2.second) ;
}

template<typename T, typename U, typename V>
const pair<T, U> operator /(const pair<T, U>& p, V value)
{
	return pair<T, U>(p.first / value, p.second / value) ;
}

template<typename T, typename U, typename V>
const pair<T, U> operator /(V value, const pair<T, U>& p)
{
	return pair<T, U>(value / p.first, value / p.second) ;
}

template<typename T, typename U>
void operator /=(pair<T, U>& p1, const pair<T, U>& p2)
{
	p1.first  /= p2.first  ;
	p1.second /= p2.second ;
}

template<typename T, typename U, typename V>
void operator /=(pair<T, U>& p, V value)
{
	p.first  /= value ;
	p.second /= value ;
}

template<typename T, typename U>
const pair<T, U> operator +(const pair<T, U>& p)
{
	return p ;
}

template<typename T, typename U>
const pair<T, U> operator -(const pair<T, U>& p)
{
	return pair<T, U>(-p.first, -p.second) ;
}
