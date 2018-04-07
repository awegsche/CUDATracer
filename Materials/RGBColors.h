#pragma once

#include "constants.h"
#include "QtGui/qimage.h"
//#include "RenderObjects/render_structs.h"

typedef float3 rgbcol;
typedef float4 rgbacol;

rgbacol Qrgb_to_float4(const QRgb col);


__device__ inline float3 rgbcolor(const float r, const float g, const float b) {
	float3 f;
	f.x = r;
	f.y = g;
	f.z = b;
	//	f.w = 1.0f;
	return f;
}
__device__ inline rgbacol rgbacolor(const float r, const float g, const float b) {
	rgbacol f;
	f.x = r;
	f.y = g;
	f.z = b;
	f.w = 1.0f;
	return f;
}

__device__ inline rgbcol rgbcolor(const rgbacol &color) {
	return rgbcolor(color.x, color.y, color.z);
}


//
//__device__ inline float4 rgbcolor(const float r, const float g, const float b, const float alpha) {
//	float4 f;
//	f.x = r;
//	f.y = g;
//	f.z = b;
//	f.w = alpha;
//	return f;
//}

//__device__ inline float4 operator-(const float4 &a, const float4 &b) {
//	return rgbcolor(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
//}

__device__ inline float3 rgbcolor(const float brightness) {
	float3 f;
	f.x = brightness;
	f.y = brightness;
	f.z = brightness;
	//f.w = 1.0f;
	return f;
}

__device__ inline uchar4 _rgbcolor_to_byte(const float3 &col) {
	uchar4 f;

	float max_channel = col.z;

	if (col.x > col.y && col.x > col.z)
		max_channel = col.x;
	else if (col.y > col.z)
		max_channel = col.y;

	float rescale = 255.0f;

	if (max_channel > 1.0f) 
		rescale = 1.0f / max_channel * 255.0f;

	f.x = (uchar)(col.x  * rescale);
	f.y = (uchar)(col.y  * rescale);
	f.z = (uchar)(col.z  * rescale);
	f.w = (uchar)255;
	
	return f;
}

__device__ inline float3 scale_color(const float3 &color, const float a) {
	return rgbcolor(color.x * a, color.y * a, color.z * a);
}

__device__ inline float3 add_colors(const float3 &a, const float3 &b) {
	return rgbcolor(a.x + b.x, a.y + b.y, a.z + b.z);
}

