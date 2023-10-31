#pragma once
#include "stdlib.h"
#include <windows.h>
#include <tchar.h>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <string>


using namespace std;

namespace dir
{
	vector<string> getSubfolders( LPCSTR path);
	vector<string> readFiles( LPCSTR path);

	vector<string> readImages(LPCSTR path);

	wstring string2wstring(const string& str);
	string    wstring2string(const wstring& wstr);

	string changeFileExt(const string& path,const string& ext);
};
