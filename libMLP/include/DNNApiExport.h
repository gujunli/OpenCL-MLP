#ifndef _DNN_API_WIN32_
#define _DNN_API_WIN32_

#ifdef _WIN32    // For Windows

#ifdef _LIBDNN_API_EXPORT_
#define LIBDNNAPI __declspec(dllexport)
#else
#define LIBDNNAPI __declspec(dllimport)
#endif

#else            // For Linux

#define LIBDNNAPI

#endif

#endif

