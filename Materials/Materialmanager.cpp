#include "Materialmanager.h"
#include <QtGui/qimage.h>
#include "RGBColors.h"
#include <cuda_runtime.h>


MaterialManager::MaterialManager()
	: texels(nullptr), positions(nullptr), dimensions(nullptr), current_texture(0)
{
}


MaterialManager::~MaterialManager()
{
}

texture_pos MaterialManager::load_texture(const QString & filename, int sprite_width, int sprite_height)
{
	QImage i;
	uint2 dims;
	dims.x = 1;
	dims.y = 1;
	
	host_positions.push_back(host_texels.size());
	

	if (i.load(filename, "png")) {

		//m_filename = filename;

		dims.x = i.width();
		dims.y = i.width();
		

		for (int x = 0; x < dims.x; x++)
			for (int y = 0; y < dims.y; y++)
			{
				QRgb col = i.pixel(x, y);
				host_texels.push_back(Qrgb_to_float4(col));
			}
		
	}
	else
	{
		
		host_texels.push_back(rgbcolor(1.0f, .0f, 1.f));
	}


	host_dims.push_back(dims);
	
	current_texture++;
	return current_texture - 1;
}

texture_pos MaterialManager::create_constantcolor(const float r, const float g, const float b)
{
	host_texels.push_back(rgbcolor(r, g, b));
	uint2 dims;
	dims.x = 1;
	dims.y = 1;
	host_dims.push_back(dims);
	
	host_positions.push_back(host_texels.size());
	

	current_texture++;
	return current_texture - 1;
}

material_pos MaterialManager::create_matte(const float k_ambient, const float k_diffuse, texture_pos position)
{
	material_params m;
	m.ka = k_ambient;
	m.kd = k_diffuse;
	m.kr = 0.0f;
	m.position = position;
	m.typ = material_type::MATTE;
	
	material_pos p = (material_pos)host_materials.size();
	host_materials.push_back(m);
	return p;
}

void MaterialManager::copy_to_device_memory()
{
	if (texels != nullptr) {
		// if device pointers have already been assigned, clean memory
		cudaFree(texels);
		cudaFree(positions);
		cudaFree(dimensions);
		cudaFree(materials);
	}

	int pos_size = sizeof(unsigned int) * host_positions.size();
	int dims_size = sizeof(uint2) *  host_dims.size();
	int materials_size = sizeof(material_params) * host_materials.size();

	cudaMalloc(&texels, sizeof(float4) * host_texels.size());
	cudaMalloc(&positions, pos_size);
	cudaMalloc(&dimensions, dims_size);
	cudaMalloc(&materials, materials_size);

	cudaMemcpy(texels, host_texels.data(), sizeof(float4) * host_texels.size(), cudaMemcpyKind::cudaMemcpyHostToDevice);
	cudaMemcpy(positions, host_positions.data(), pos_size, cudaMemcpyKind::cudaMemcpyHostToDevice);
	cudaMemcpy(dimensions, host_dims.data(), dims_size, cudaMemcpyKind::cudaMemcpyHostToDevice);
	cudaMemcpy(materials, host_materials.data(), materials_size, cudaMemcpyKind::cudaMemcpyHostToDevice);


	//int tex_pos = 0;
	//int pos_pos = 0;
	//int dim_pos = 0;
	//int mat_pos = 0;

	//for (int i = 0; i < host_materials.size(); i++) {

	//	cudaMemcpy(texels + tex_pos, &host_texels[i], sizeof(float4), cudaMemcpyKind::cudaMemcpyHostToDevice);
	//	cudaMemcpy(positions + pos_pos, &host_positions[i], sizeof(uint), cudaMemcpyKind::cudaMemcpyHostToDevice);
	//	cudaMemcpy(dimensions + dim_pos, &host_dims[i], sizeof(uint2), cudaMemcpyKind::cudaMemcpyHostToDevice);
	//	cudaMemcpy(materials + mat_pos, &host_materials[i], sizeof(material_params), cudaMemcpyKind::cudaMemcpyHostToDevice);

	//	tex_pos += sizeof(float4);
	//	pos_pos += sizeof(uint);
	//	dim_pos += sizeof(uint2);
	//	mat_pos += sizeof(material_params);
	//}

}

