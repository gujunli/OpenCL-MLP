#ifndef _MLP_API_WIN32_
#define _MLP_API_WIN32_

#ifdef _WIN32    // For Windows

#ifdef _LIBMLP_API_EXPORT_
#define LIBMLPAPI __declspec(dllexport)
#else
#define LIBMLPAPI __declspec(dllimport)
#endif

#else            // For Linux

#define LIBMLPAPI 

#endif 

#endif 

