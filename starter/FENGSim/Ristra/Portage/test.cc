#include "test.h"

void export_vtk (int n, std::string filename, std::vector<double> val) {
    std::ofstream out(filename);
    out << "# vtk DataFile Version 2.0" << std::endl;
    out << "Structured Grid by Portage" << std::endl;
    out << "ASCII" << std::endl;
    out << "DATASET UNSTRUCTURED_GRID" << std::endl;
    out << "POINTS " << 4*n*n << " float" << std::endl;
    double d = 1.0 / n;
    for (int i=0; i<n; i++) {
        for (int j=0; j<n; j++) {
	    out << i*d << " " << j*d << " " << 0 << std::endl;
	    out << (i+1)*d << " " << j*d << " " << 0 << std::endl;
	    out << (i+1)*d << " " << (j+1)*d << " " << 0 << std::endl;
	    out << i*d << " " << (j+1)*d << " " << 0 << std::endl; 
	}
    }
    out << "CELLS " << n*n << " "<< n*n*5  << std::endl;
    for (int i=0; i<n*n; i++) {
        out << "4 " << i*4 << " " << i*4 + 1 << " " << i*4 + 2 << " " << i*4 + 3 << std::endl;
    }
    out << "CELL_TYPES " << n*n << std::endl;
    for (int i=0; i<n*n; i++) {
        out << "9" << std::endl;
    }
    out << "CELL_DATA " << n*n << std::endl;
    out << "SCALARS cell_scalars float" << std::endl;
    out << "LOOKUP_TABLE default" << std::endl;
    for (int i=0; i<n*n; i++) {
        out << val[i] << std::endl;
    }
    out.close();
}

void export_overlap (const Wonton::Simple_Mesh_Wrapper& source, std::vector<int> candidates,
					 const Wonton::Simple_Mesh_Wrapper& target, int cell_id) {
	std::ofstream out("target_cell.vtk");
    out << "# vtk DataFile Version 2.0" << std::endl;
    out << "Structured Grid by Portage" << std::endl;
    out << "ASCII" << std::endl;
    out << "DATASET UNSTRUCTURED_GRID" << std::endl;
    out << "POINTS " << 4 << " float" << std::endl;
	std::vector<Wonton::Point<2>> cell_coord;
	target.cell_get_coordinates(cell_id, &cell_coord);
	for (int j=0; j<cell_coord.size(); j++)
		out << cell_coord[j] << " 0" << std::endl;
    
    out << "CELLS " << 1 << " "<< 5  << std::endl;
	out << "4 " << 0 << " " << 1 << " " << 2 << " " << 3 << std::endl;
	
    out << "CELL_TYPES " << 1 << std::endl;
	out << "9" << std::endl;
	
    out << "CELL_DATA " << 1 << std::endl;
    out << "SCALARS cell_scalars float" << std::endl;
    out << "LOOKUP_TABLE default" << std::endl;
	out << 1 << std::endl;
    out.close();

    out.open("target_mesh.vtk");
    out << "# vtk DataFile Version 2.0" << std::endl;
    out << "Structured Grid by Portage" << std::endl;
    out << "ASCII" << std::endl;
    out << "DATASET UNSTRUCTURED_GRID" << std::endl;
	int target_cell_num = target.num_entities(Wonton::Entity_kind::CELL, Wonton::Entity_type::ALL);
    out << "POINTS " << 4*target_cell_num << " float" << std::endl;
    for (int i=0; i<target_cell_num; i++) {
		target.cell_get_coordinates(i, &cell_coord);
		for (int j=0; j<cell_coord.size(); j++)
			out << cell_coord[j] << " 0" << std::endl;
    }
    out << "CELLS " << target_cell_num << " "<< target_cell_num*5  << std::endl;
    for (int i=0; i<target_cell_num; i++) {
        out << "4 " << i*4 << " " << i*4 + 1 << " " << i*4 + 2 << " " << i*4 + 3 << std::endl;
    }
    out << "CELL_TYPES " << target_cell_num << std::endl;
    for (int i=0; i<target_cell_num; i++) {
        out << "9" << std::endl;
    }
    out << "CELL_DATA " << target_cell_num << std::endl;
    out << "SCALARS cell_scalars float" << std::endl;
    out << "LOOKUP_TABLE default" << std::endl;
    for (int i=0; i<target_cell_num; i++) {
		if (i==cell_id) {
			out << 1 << std::endl;
			continue;
		}
        out << 0 << std::endl;
    }
    out.close();


	out.open("source_mesh.vtk");
    out << "# vtk DataFile Version 2.0" << std::endl;
    out << "Structured Grid by Portage" << std::endl;
    out << "ASCII" << std::endl;
    out << "DATASET UNSTRUCTURED_GRID" << std::endl;
	int source_cell_num = source.num_entities(Wonton::Entity_kind::CELL, Wonton::Entity_type::ALL);
    out << "POINTS " << 4*source_cell_num << " float" << std::endl;
    for (int i=0; i<source_cell_num; i++) {
		source.cell_get_coordinates(i, &cell_coord);
		for (int j=0; j<cell_coord.size(); j++)
			out << cell_coord[j] << " 0" << std::endl;
    }
    out << "CELLS " << source_cell_num << " "<< source_cell_num*5  << std::endl;
    for (int i=0; i<source_cell_num; i++) {
        out << "4 " << i*4 << " " << i*4 + 1 << " " << i*4 + 2 << " " << i*4 + 3 << std::endl;
    }
    out << "CELL_TYPES " << source_cell_num << std::endl;
    for (int i=0; i<source_cell_num; i++) {
        out << "9" << std::endl;
    }
    out << "CELL_DATA " << source_cell_num << std::endl;
    out << "SCALARS cell_scalars float" << std::endl;
    out << "LOOKUP_TABLE default" << std::endl;
    for (int i=0; i<source_cell_num; i++) {
		bool check = false;
		for (int j=0; j<candidates.size(); j++) {
			if (i==candidates[j]) {
				check = true;
			}
		}
		if (check) {
			out << 1 << std::endl;
			continue;
		}
        out << 0 << std::endl;
    }
    out.close();
}

void export_vtk (const Wonton::Jali_Mesh_Wrapper& source) {
	int source_cell_num = source.num_entities(Wonton::Entity_kind::CELL, Wonton::Entity_type::ALL);
	int m = 0;
    for (int i=0; i<source_cell_num; i++) {
		if (source.cell_get_element_type(i) == 1) {
			m+=3;
		}
		if (source.cell_get_element_type(i) == 2) {
			m+=4;
		}
    }
	std::ofstream out("mesh.vtk");
    out << "# vtk DataFile Version 2.0" << std::endl;
    out << "Structured Grid by Portage" << std::endl;
    out << "ASCII" << std::endl;
    out << "DATASET UNSTRUCTURED_GRID" << std::endl;
    out << "POINTS " << m << " float" << std::endl;
	std::vector<Wonton::Point<2>> cell_coord;
    for (int i=0; i<source_cell_num; i++) {
		source.cell_get_coordinates(i, &cell_coord);
		for (int j=0; j<cell_coord.size(); j++)
			out << cell_coord[j] << " 0" << std::endl;
    }
    out << "CELLS " << source_cell_num << " "<< source_cell_num+m  << std::endl;
	int n = 0;
    for (int i=0; i<source_cell_num; i++) {
		if (source.cell_get_element_type(i) == 1) {
			out << "3 " << n << " " << n + 1 << " " << n + 2 << std::endl;
			n+=3;
		}
		if (source.cell_get_element_type(i) == 2) {
			out << "4 " << n << " " << n + 1 << " " << n + 2 << " " << n + 3 << std::endl;
			n+=4;
		}
    }
    out << "CELL_TYPES " << source_cell_num << std::endl;
    for (int i=0; i<source_cell_num; i++) {
		if (source.cell_get_element_type(i) == 1)
		    out << "5" << std::endl;
		if (source.cell_get_element_type(i) == 2)
		    out << "9" << std::endl;
    }
    out.close();

}


void export_overlap (const Wonton::Jali_Mesh_Wrapper& source, std::vector<int> candidates,
					 const Wonton::Jali_Mesh_Wrapper& target, int cell_id) {
	std::ofstream out("target_cell.vtk");
    out << "# vtk DataFile Version 2.0" << std::endl;
    out << "Structured Grid by Portage" << std::endl;
    out << "ASCII" << std::endl;
    out << "DATASET UNSTRUCTURED_GRID" << std::endl;
    out << "POINTS " << 3 << " float" << std::endl;
	std::vector<Wonton::Point<2>> cell_coord;
	target.cell_get_coordinates(cell_id, &cell_coord);
	for (int j=0; j<cell_coord.size(); j++)
		out << cell_coord[j] << " 0" << std::endl;
    
    out << "CELLS " << 1 << " "<< 4  << std::endl;
	out << "3 " << 0 << " " << 1 << " " << 2 << std::endl;
	
    out << "CELL_TYPES " << 1 << std::endl;
	out << "5" << std::endl;
	
    out << "CELL_DATA " << 1 << std::endl;
    out << "SCALARS cell_scalars float" << std::endl;
    out << "LOOKUP_TABLE default" << std::endl;
	out << 1 << std::endl;
    out.close();

    out.open("target_mesh.vtk");
    out << "# vtk DataFile Version 2.0" << std::endl;
    out << "Structured Grid by Portage" << std::endl;
    out << "ASCII" << std::endl;
    out << "DATASET UNSTRUCTURED_GRID" << std::endl;
	int target_cell_num = target.num_entities(Wonton::Entity_kind::CELL, Wonton::Entity_type::ALL);
    out << "POINTS " << 3*target_cell_num << " float" << std::endl;
    for (int i=0; i<target_cell_num; i++) {
		target.cell_get_coordinates(i, &cell_coord);
		for (int j=0; j<cell_coord.size(); j++)
			out << cell_coord[j] << " 0" << std::endl;
    }
    out << "CELLS " << target_cell_num << " "<< target_cell_num*4  << std::endl;
    for (int i=0; i<target_cell_num; i++) {
        out << "3 " << i*3 << " " << i*3 + 1 << " " << i*3 + 2 << std::endl;
    }
    out << "CELL_TYPES " << target_cell_num << std::endl;
    for (int i=0; i<target_cell_num; i++) {
        out << "5" << std::endl;
    }
    out << "CELL_DATA " << target_cell_num << std::endl;
    out << "SCALARS cell_scalars float" << std::endl;
    out << "LOOKUP_TABLE default" << std::endl;
    for (int i=0; i<target_cell_num; i++) {
		if (i==cell_id) {
			out << 1 << std::endl;
			continue;
		}
        out << 0 << std::endl;
    }
    out.close();


	out.open("source_mesh.vtk");
    out << "# vtk DataFile Version 2.0" << std::endl;
    out << "Structured Grid by Portage" << std::endl;
    out << "ASCII" << std::endl;
    out << "DATASET UNSTRUCTURED_GRID" << std::endl;
	int source_cell_num = source.num_entities(Wonton::Entity_kind::CELL, Wonton::Entity_type::ALL);
    out << "POINTS " << 3*source_cell_num << " float" << std::endl;
    for (int i=0; i<source_cell_num; i++) {
		source.cell_get_coordinates(i, &cell_coord);
		for (int j=0; j<cell_coord.size(); j++)
			out << cell_coord[j] << " 0" << std::endl;
    }
    out << "CELLS " << source_cell_num << " "<< source_cell_num*4  << std::endl;
    for (int i=0; i<source_cell_num; i++) {
        out << "3 " << i*3 << " " << i*3 + 1 << " " << i*3 + 2 << std::endl;
    }
    out << "CELL_TYPES " << source_cell_num << std::endl;
    for (int i=0; i<source_cell_num; i++) {
        out << "5" << std::endl;
    }
    out << "CELL_DATA " << source_cell_num << std::endl;
    out << "SCALARS cell_scalars float" << std::endl;
    out << "LOOKUP_TABLE default" << std::endl;
    for (int i=0; i<source_cell_num; i++) {
		bool check = false;
		for (int j=0; j<candidates.size(); j++) {
			if (i==candidates[j]) {
				check = true;
			}
		}
		if (check) {
			out << 1 << std::endl;
			continue;
		}
        out << 0 << std::endl;
    }
    out.close();
}

void export_vtk (std::vector<std::vector<Wonton::Point<2>>>& mesh, std::vector<double>& cellvecout) {
	int cell_num = mesh.size();
    std::ofstream out("remapping.vtk");
    out << "# vtk DataFile Version 2.0" << std::endl;
    out << "Structured Grid by Portage" << std::endl;
    out << "ASCII" << std::endl;
    out << "DATASET UNSTRUCTURED_GRID" << std::endl;
    out << "POINTS " << 3*cell_num << " float" << std::endl;
	for (int c = 0; c < mesh.size(); ++c) {
		for (int i=0; i<3; i++)
			out << mesh[c][i] << " " << 0 << std::endl;
	}
    out << "CELLS " << cell_num << " "<< 4*cell_num  << std::endl;
    for (int i=0; i<cell_num; i++) {
        out << "3 " << i*3 << " " << i*3 + 1 << " " << i*3 + 2 << std::endl;
    }
    out << "CELL_TYPES " << cell_num << std::endl;
    for (int i=0; i<cell_num; i++) {
        out << "5" << std::endl;
    }
    out << "CELL_DATA " << cell_num << std::endl;
    out << "SCALARS cell_scalars float" << std::endl;
    out << "LOOKUP_TABLE default" << std::endl;
    for (int i=0; i<cell_num; i++) {
        out << cellvecout[i] << std::endl;
    }
    out.close();

}

void export_vtk (Jali::UniStateVector<double, Jali::Mesh> cellvecout, const Wonton::Jali_Mesh_Wrapper& targetMeshWrapper) {
	const int ntarcells = targetMeshWrapper.num_owned_cells();
	std::vector<double> val;
	val.resize(ntarcells);
	for (int c = 0; c < ntarcells; c++) {
		val[c] = cellvecout[c];
	}
	std::vector<std::vector<Wonton::Point<2>>> tar_mesh;
	for (int c = 0; c < ntarcells; c++) {
		std::vector<Wonton::Point<2>> cell_coords;
		targetMeshWrapper.cell_get_coordinates(c, &cell_coords);
		tar_mesh.push_back(cell_coords);
	}
	export_vtk(tar_mesh, val);

}
