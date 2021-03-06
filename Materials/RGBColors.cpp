#include "RGBColors.h"



rgbacol Qrgb_to_float4(const QRgb col)
{
	rgbacol f;
	f.z = (float)((col & 0x000000FF)) / 255.0f;
	f.y = (float)((col & 0x0000FF00) >> 8) / 255.0f;
	f.x = (float)((col & 0x00FF0000) >> 16) / 255.0f;
	f.w = (float)((col & 0xFF000000) >> 24) / 255.0f;

	return f;
}
