#include "dir.h"

//------------------------------------------------------------------------------------
// function to load all sub folders for a given directory
//------------------------------------------------------------------------------------
vector<string> dir::getSubfolders(LPCSTR path)
{
	// make sure that the path has an ok format
	string Path(path);
	if(Path[Path.size()-1]!='*')
	{
		if(Path[Path.size()-1]!='\\')
			Path.push_back('\\');
		Path.push_back('*');
	}
	vector<string> subfolders;
   WIN32_FIND_DATA FindFileData;
   HANDLE hFind;
	
   hFind = FindFirstFile(Path.data(), &FindFileData);
   if (hFind == INVALID_HANDLE_VALUE) 
   {
      printf ("FindFirstFile failed (%d)\n", GetLastError());
      return subfolders;
   } 
   bool IsFound;
   do{
	   if((FindFileData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)== FILE_ATTRIBUTE_DIRECTORY &&
		   (FindFileData.dwFileAttributes & FILE_ATTRIBUTE_SYSTEM) != FILE_ATTRIBUTE_SYSTEM)
	   {
		   if(strcmp(FindFileData.cFileName,".") != 0 && strcmp(FindFileData.cFileName,"..") != 0)
		   {
			   //wcout<<FindFileData.dwFileAttributes << "\t" << FindFileData.cFileName <<endl;
			   subfolders.push_back(string(FindFileData.cFileName));
		   }
	   }
	   
	   IsFound = FindNextFile(hFind,&FindFileData);
   }while(IsFound);
   FindClose(hFind);
   return subfolders;
}

//------------------------------------------------------------------------------------
// function to load all sub folders for a given directory
//------------------------------------------------------------------------------------
vector<string> dir::readFiles(LPCSTR path)
{
	vector<string> subfolders;
   WIN32_FIND_DATA FindFileData;
   HANDLE hFind;
	
   hFind = FindFirstFile(path, &FindFileData);
   if (hFind == INVALID_HANDLE_VALUE) 
   {
      //printf ("FindFirstFile failed (%d)\n", GetLastError());
      return subfolders;
   } 
   bool IsFound;
   do{
	   subfolders.push_back(string(FindFileData.cFileName));
	   //wcout<<FindFileData.dwFileAttributes << "\t" << FindFileData.cFileName <<endl;
	   IsFound = FindNextFile(hFind,&FindFileData);
   }while(IsFound);
   FindClose(hFind);
   return subfolders;
}

//------------------------------------------------------------------------------------
// function to load all images in a given folder
//------------------------------------------------------------------------------------
vector<string> dir::readImages(LPCSTR path)
{
	string Path(path);
	if(Path[Path.size()-1]!='\\')
		Path.push_back('\\');

	vector<string> filelist;
	LPCSTR imgformats[4]={"*.jpg","*.jpeg","*.bmp","*.png"};//,L"*.gif"};
	for(int  i=0;i<4;i++)
	{
		vector<string> templist;
		string search(Path);
		search+= imgformats[i];
		//wcout<<search.data() <<endl;
		templist = readFiles(search.data());
		
		for(int j = 0;j<templist.size(); j++)
			filelist.push_back(templist[j]);
	}
	return filelist;
}

//------------------------------------------------------------------------------------
// function to convert wstring to string
//------------------------------------------------------------------------------------
string dir::wstring2string(const wstring& wstr)
{
	string s(wstr.begin(),wstr.end());
	s.assign(wstr.begin(),wstr.end());
	return s;
}

wstring dir::string2wstring(const string& str)
{
	wstring ws(str.begin(),str.end());
	ws.assign(str.begin(),str.end());
	return ws;
}

//------------------------------------------------------------------------------------
// function to change file extension
//------------------------------------------------------------------------------------
string dir::changeFileExt(const string& path,const string& ext)
{
	string newpath;
	int dot = path.find_last_of('.');
	newpath = path.substr(0,dot+1);
	newpath+=ext;
	return newpath;
}