/**
 * 以下一堆不知道什么东西来源于
 * https://github.com/ShiqiYu/libfacedetection/issues/222
 * 虽然我觉得可能直接删掉就行了
 */
#pragma once

#if defined(_WIN32) && defined(FACEDETECTION_BUILD)
#define FACEDETECTION_EXPORT __declspec(dllexport)
#elif defined(_WIN32)
#define FACEDETECTION_EXPORT __declspec(dllimport)
#else
#define FACEDETECTION_EXPORT
#endif