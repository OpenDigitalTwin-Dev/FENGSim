//Copyright (c) 2019 Ultimaker B.V.
//CuraEngine is released under the terms of the AGPLv3 or higher.

#include <gtest/gtest.h>

#include "../../../../../../toolkit/cura_engine/src/infill.h"
#include "ReadTestPolygons.h"

//#define TEST_INFILL_SVG_OUTPUT
#ifdef TEST_INFILL_SVG_OUTPUT
#include <cstdlib>
#include "../src/utils/SVG.h"
#endif //TEST_INFILL_SVG_OUTPUT
#include "fstream"

namespace cura
{
    template<typename ... Ts>
    std::string makeName(const std::string& format_string, Ts ... args)
    {
        constexpr int buff_size = 1024;
        char buff[buff_size];
        std::snprintf(buff, buff_size, format_string.c_str(), args...);
        return std::string(buff);
    }
	
    coord_t getPatternMultiplier(const EFillMethod& pattern)
    {
        switch (pattern)
			{
			case EFillMethod::GRID: // fallthrough
			case EFillMethod::TETRAHEDRAL: // fallthrough
			case EFillMethod::QUARTER_CUBIC:
				return 2;
			case EFillMethod::TRIANGLES: // fallthrough
			case EFillMethod::TRIHEXAGON: // fallthrough
			case EFillMethod::CUBIC: // fallthrough
			case EFillMethod::CUBICSUBDIV:
				return 3;
			default:
				return 1;
			}
    }
  
    struct InfillParameters
    {
    public:
        // Actual infill parameters:
        EFillMethod pattern;
        bool zig_zagify;
        bool connect_polygons;
        coord_t line_distance;
		
        std::string name;
	
        InfillParameters(const EFillMethod& pattern, const bool& zig_zagify, const bool& connect_polygons, const coord_t& line_distance) :
            pattern(pattern),
            zig_zagify(zig_zagify),
            connect_polygons(connect_polygons),
            line_distance(line_distance)
	    {
		name = makeName("InfillParameters_%d_%d_%d_%lld", (int)pattern, (int)zig_zagify, (int)connect_polygons, line_distance);
	    }
    };
    
    class InfillTestParameters
    {
    public:
        bool valid;  // <-- if the file isn't read (or anything else goes wrong with the setup) we can communicate it to the tests
        std::string fail_reason;
        size_t test_polygon_id;
		
        // Parameters used to generate the infill:
        InfillParameters params;
        Polygons outline_polygons;
		
        // Resulting infill:
        Polygons result_lines;
        Polygons result_polygons;
		
        std::string name;
		
        InfillTestParameters() :
            valid(false),
            fail_reason("Read of file with test polygons failed (see generateInfillTests), can't continue tests."),
            test_polygon_id(-1),
            params(InfillParameters(EFillMethod::NONE, false, false, 0)),
            outline_polygons(Polygons()),
            result_lines(Polygons()),
            result_polygons(Polygons()),
            name("UNNAMED")
	    {
	    }
		
        InfillTestParameters(const InfillParameters& params, const size_t& test_polygon_id, const Polygons& outline_polygons, const Polygons& result_lines, const Polygons& result_polygons) :
            valid(true),
            fail_reason("__"),
            test_polygon_id(test_polygon_id),
            params(params),
            outline_polygons(outline_polygons),
            result_lines(result_lines),
            result_polygons(result_polygons)
	    {
		name = makeName("InfillTestParameters_P%d_Z%d_C%d_L%lld__%lld", (int)params.pattern, (int)params.zig_zagify, (int)params.connect_polygons, params.line_distance, test_polygon_id);
	    }
		
        friend std::ostream& operator<<(std::ostream& os, const InfillTestParameters& params)
	    {
		return os << params.name << "(" << (params.valid ? std::string("input OK") : params.fail_reason) << ")";
	    }
    };

    constexpr coord_t outline_offset = 0;
    constexpr coord_t infill_line_width = 350;
    constexpr coord_t infill_overlap = 0;
    constexpr size_t infill_multiplier = 1;
    const AngleDegrees fill_angle = 0.;
    constexpr coord_t z = 100; // Future improvement: Also take an uneven layer, so we get the alternate.
    constexpr coord_t shift = 0;
    const std::vector<std::string> polygon_filenames =
    {
        "/home/jiping/OpenDT/CAM/Cura/Cura/conf/geo/tests/resources/polygon_concave.txt",
        "/home/jiping/OpenDT/CAM/Cura/Cura/conf/geo/tests/resources/polygon_concave_hole.txt",
        "/home/jiping/OpenDT/CAM/Cura/Cura/conf/geo/tests/resources/polygon_square.txt",
        "/home/jiping/OpenDT/CAM/Cura/Cura/conf/geo/tests/resources/polygon_square_hole.txt",
        "/home/jiping/OpenDT/CAM/Cura/Cura/conf/geo/tests/resources/polygon_triangle.txt",
        "/home/jiping/OpenDT/CAM/Cura/Cura/conf/geo/tests/resources/polygon_two_squares.txt",
    };
	
    InfillTestParameters generateInfillToTest(const InfillParameters& params, const size_t& test_polygon_id, const Polygons& outline_polygons)
    {
        const EFillMethod pattern = params.pattern;
        const bool zig_zagify = params.zig_zagify;
        const bool connect_polygons = params.connect_polygons;
        const coord_t line_distance = params.line_distance;

        Infill infill
	    (
		pattern,
		zig_zagify,
		connect_polygons,
		outline_polygons,
		outline_offset,
		infill_line_width,
		line_distance,
		infill_overlap,
		infill_multiplier,
		fill_angle,
		z,
		shift
		); // There are some optional parameters, but these will do for now (future improvement?).

        Polygons result_polygons;
        Polygons result_lines;
        infill.generate(result_polygons, result_lines, nullptr, nullptr);

        InfillTestParameters result = InfillTestParameters(params, test_polygon_id, outline_polygons, result_lines, result_polygons);
        return result;
    }

    std::vector<Point3> pnts;
    std::vector<double> heights;
    
    void ExportOutLinesToVtk (Polygons P, double height, std::string filename=" ") {
	int n = 0;
	for (int i = 0; i < P.size(); i++) {
	    n += P[i].size();
	}
	
	std::ofstream out;
	out.open((std::string("./data/vtk/")+filename).c_str());
	//out.open(filename.c_str());
	
	out <<"# vtk DataFile Version 2.0" << std::endl;
	out << "slices example" << std::endl;
	out << "ASCII" << std::endl;
	out << "DATASET UNSTRUCTURED_GRID" << std::endl;
	out << "POINTS " << n << " float" << std::endl;
	for (int i = 0; i < P.size(); i++) {
	    for (int j = 0; j < P[i].size(); j++) {
		out << P[i][j].X << " " << P[i][j].Y << " " << height << std::endl;
	    }
	}
	out << "CELLS " << P.size() << " " << P.size() + n + P.size() << std::endl;
	int m = 0;
	for (int i = 0; i < P.size(); i++) {
	    out << P[i].size() + 1;
	    for(int j = 0; j < P[i].size(); j++) { //Find the starting corner in the sliced layer.
		out << " " << m;
		m++;
	    }
	    out << " " << m - P[i].size();
	    out << std::endl;
	}
	out << "CELL_TYPES " << P.size() << std::endl;
	for (int i = 0; i < P.size(); i++) {
	    out << 4 << std::endl;
	}
    }
  
    void ExportPathLinesToVtk (Polygons P, double height, std::string filename) {
	int n = 0;
	for (int i = 0; i < P.size(); i++) {
	    n += P[i].size();
	}
	
	std::ofstream out;
	out.open((std::string("./data/vtk/")+filename).c_str());
	//out.open(filename.c_str());
	
	out <<"# vtk DataFile Version 2.0" << std::endl;
	out << "slices example" << std::endl;
	out << "ASCII" << std::endl;
	out << "DATASET UNSTRUCTURED_GRID" << std::endl;
	out << "POINTS " << n << " float" << std::endl;
	for (int i = 0; i < P.size(); i++) {
	    for (int j = 0; j < P[i].size(); j++) {
		out << P[i][j].X << " " << P[i][j].Y << " " << height << std::endl;
	    }
	}
	out << "CELLS " << P.size() << " " << P.size() + n << std::endl;
	int m = 0;
	for (int i = 0; i < P.size(); i++) {
	    out << P[i].size();
	    for(int j = 0; j < P[i].size(); j++) { //Find the starting corner in the sliced layer.
		out << " " << m;
		m++;
	    }
	    out << std::endl;
	}
	out << "CELL_TYPES " << P.size() << std::endl;
	for (int i = 0; i < P.size(); i++) {
	    out << 4 << std::endl;
	}
    }

    std::vector<InfillTestParameters> generateInfillTests()
    {
        constexpr bool do_zig_zaggify = true;
        constexpr bool dont_zig_zaggify = false;
        constexpr bool do_connect_polygons = true;
        constexpr bool dont_connect_polygons = false;
	
        std::vector<Polygons> shapes;
        if (!readTestPolygons(polygon_filenames, shapes))
	    {
		return { InfillTestParameters() };  // return an invalid singleton, that'll trip up the 'file read' assertion in the TEST_P's
	    }
	
        /* Skip methods:
	   - that require the SierpinskyInfillProvider class, since these test classes aren't equipped to handle that yet
	   this can be considered a TODO for these testcases here, not in the methods themselves
	   (these are; Cross, Cross-3D and Cubic-Subdivision)
	   - Gyroid, since it doesn't handle the 100% infill and related cases well
	*/
        std::vector<EFillMethod> skip_methods = { EFillMethod::CROSS, EFillMethod::CROSS_3D, EFillMethod::CUBICSUBDIV, EFillMethod::GYROID };

        std::vector<EFillMethod> methods;
        for (int i_method = 0; i_method < static_cast<int>(EFillMethod::NONE); ++i_method)
	    {
		const EFillMethod method = static_cast<EFillMethod>(i_method);
		if (std::find(skip_methods.begin(), skip_methods.end(), method) == skip_methods.end()) // Only use if not in skipped.
		    {
			methods.push_back(method);
		    }
	    }

        std::vector<coord_t> line_distances = { 350, 400, 600, 800, 1200 };

        std::vector<InfillTestParameters> parameters_list;
        size_t test_polygon_id = 0;
        for (const Polygons& polygons : shapes)
	    {
		for (const EFillMethod& method : methods)
		    {
			for (const coord_t& line_distance : line_distances)
			    {
				parameters_list.push_back(generateInfillToTest(InfillParameters(method, dont_zig_zaggify, dont_connect_polygons, line_distance), test_polygon_id, polygons));
				parameters_list.push_back(generateInfillToTest(InfillParameters(method, dont_zig_zaggify, do_connect_polygons, line_distance), test_polygon_id, polygons));
				parameters_list.push_back(generateInfillToTest(InfillParameters(method, do_zig_zaggify, dont_connect_polygons, line_distance), test_polygon_id, polygons));
				parameters_list.push_back(generateInfillToTest(InfillParameters(method, do_zig_zaggify, do_connect_polygons, line_distance), test_polygon_id, polygons));
			    }
		    }
		++test_polygon_id;
	    }

	//ExportOutLinesToVtk(parameters_list[1].outline_polygons, "infill_outlines_nonzig.vtk");
	//ExportPathLinesToVtk(parameters_list[1].result_lines, "infill_pathlines_nonzig.vtk");
	//ExportOutLinesToVtk(parameters_list[2].outline_polygons, "infill_outlines_zig.vtk");
	//ExportPathLinesToVtk(parameters_list[2].result_lines, "infill_pathlines_zig.vtk");
	
	return parameters_list;
    }
    
    void VtkToPolygons (std::string filename, std::vector<Polygons>& layers) {
	heights.clear();
	pnts.clear();
	
	std::ifstream is;
	is.open(filename.c_str());

	const int len = 512;
	char L[len];

	for (int i = 0; i < 5; i++) is.getline(L,len);

	double scale = 1000;
	
	while (is.getline(L,len)) {
	    if (strncasecmp("POLYGONS", L, 8) == 0) break;
	    double z[3];
	    sscanf(L, "%lf %lf %lf", z, z + 1, z + 2);
	    Point3 p;
	    p.x = z[0] * scale;
	    p.y = z[1] * scale;
	    p.z = z[2] * scale;
	    pnts.push_back(p);

	    if (heights.size() > 0) {
	        if ((z[2] * scale) != heights[heights.size()-1])
		    heights.push_back(z[2] * scale);
	    }
	    else {
		heights.push_back(z[2] * scale);
	    }
	}
	
	while (is.getline(L,len)) {
	    cura::Polygon Poly;
	    cura::Polygons Polys;
	    
	    int m = -1;
	    sscanf(L, "%d[^ ]", &m);

	    std::string ss1 = "%*d";
	    for (int i = 0; i < m; i++) {
		std::string ss2 = "";
		ss2 = ss1 + " %d[^ ]";
		int v;
		sscanf(L, ss2.c_str(), &v);
		Poly.emplace_back(pnts[v].x, pnts[v].y);
		ss1 += " %*d";
	    }
	    Polys.add(Poly);
	    layers.push_back(cura::Polygons(Polys));
	}
	std::cout << "layers: " << layers.size() << std::endl;
	is.close();
    }

    std::vector<InfillTestParameters> generateInfillTests(std::string filename)
    {
        constexpr bool do_zig_zaggify = true;
        constexpr bool dont_zig_zaggify = false;
        constexpr bool do_connect_polygons = true;
        constexpr bool dont_connect_polygons = false;

	std::ifstream is;
	is.open(std::string("./solver/conf/cura.conf").c_str());
	const int len = 512;
	char L[len];
	is.getline(L,len);
	is.getline(L,len);
	std::string clifile = L;
	is.getline(L,len);
	std::string pathfile = L;

	std::cout << clifile << std::endl;
	std::cout << pathfile << std::endl;
	is.close();





	
        std::vector<Polygons> shapes;
	//VtkToPolygons("/home/jiping/M++/data/vtk/slices.vtk", shapes);
	VtkToPolygons(clifile.c_str(), shapes);







	
	/*
	  if (!readTestPolygons(polygon_filenames, shapes))
	  {
	  return { InfillTestParameters() };  // return an invalid singleton, that'll trip up the 'file read' assertion in the TEST_P's
	  }*/
	
        /* Skip methods:
	   - that require the SierpinskyInfillProvider class, since these test classes aren't equipped to handle that yet
	   this can be considered a TODO for these testcases here, not in the methods themselves
	   (these are; Cross, Cross-3D and Cubic-Subdivision)
	   - Gyroid, since it doesn't handle the 100% infill and related cases well
	*/
        std::vector<EFillMethod> skip_methods = { EFillMethod::CROSS, EFillMethod::CROSS_3D, EFillMethod::CUBICSUBDIV, EFillMethod::GYROID };

        std::vector<EFillMethod> methods;
        for (int i_method = 0; i_method < static_cast<int>(EFillMethod::NONE); ++i_method)
	    {
		const EFillMethod method = static_cast<EFillMethod>(i_method);
		if (std::find(skip_methods.begin(), skip_methods.end(), method) == skip_methods.end()) // Only use if not in skipped.
		    {
			methods.push_back(method);
		    }
	    }

        //std::vector<coord_t> line_distances = { 350, 400, 600, 800, 1200 };
	std::vector<coord_t> line_distances = { 500, 400, 600, 800, 1200 };
	
        std::vector<InfillTestParameters> parameters_list;
        size_t test_polygon_id = 0;
        for (const Polygons& polygons : shapes) {
	    //for (const EFillMethod& method : methods) {
	    //for (const coord_t& line_distance : line_distances) {
	    //parameters_list.push_back(generateInfillToTest(InfillParameters(methods[0], dont_zig_zaggify, dont_connect_polygons, line_distances[0]), test_polygon_id, polygons));
	    //parameters_list.push_back(generateInfillToTest(InfillParameters(method, dont_zig_zaggify, do_connect_polygons, line_distance), test_polygon_id, polygons));
	    parameters_list.push_back(generateInfillToTest(InfillParameters(methods[0], do_zig_zaggify, dont_connect_polygons, line_distances[0]), test_polygon_id, polygons));
	    //parameters_list.push_back(generateInfillToTest(InfillParameters(method, do_zig_zaggify, do_connect_polygons, line_distance), test_polygon_id, polygons));
	    //}
	    //}
	    ++test_polygon_id;
        }

	for (int i = 0; i < parameters_list.size(); i++) {
	    //ExportOutLinesToVtk(parameters_list[i].outline_polygons, heights[i], "infill_outlines"+std::to_string(i)+".vtk");
	    //ExportPathLinesToVtk(parameters_list[i].result_lines, heights[i], "infill_pathlines"+std::to_string(i)+".vtk");
	    //ExportOutLinesToVtk(parameters_list[i].outline_polygons, heights[i], pathfile+"_outlines"+std::to_string(i)+".vtk");
	    //ExportPathLinesToVtk(parameters_list[i].result_lines, heights[i], pathfile+"_pathlines"+std::to_string(i)+".vtk");
	}




	// collect all path lines together



	

	double scale = 1000;
	
	int n = 0;
	for (int i = 0; i < parameters_list.size(); i++) {
	    for (int j = 0; j < parameters_list[i].result_lines.size(); j++) {
	        n += (parameters_list[i].result_lines)[j].size();
	    }
	}
	int m = 0;
	for (int i = 0; i < parameters_list.size(); i++) {
	    for (int j = 0; j < parameters_list[i].result_lines.size(); j++) {
		m++;
	    }
	}

	std::ofstream out;
	out.open(pathfile);
	
	out <<"# vtk DataFile Version 2.0" << std::endl;
	out << "slices example" << std::endl;
	out << "ASCII" << std::endl;
	out << "DATASET UNSTRUCTURED_GRID" << std::endl;
	out << "POINTS " << n << " float" << std::endl;
	for (int i = 0; i < parameters_list.size(); i++) {
	    for (int j = 0; j < parameters_list[i].result_lines.size(); j++) {
		for (int k = 0; k < (parameters_list[i].result_lines)[j].size(); k++) {
		    out << (parameters_list[i].result_lines)[j][k].X / scale << " " << (parameters_list[i].result_lines)[j][k].Y / scale << " " << heights[i] / scale << std::endl;
		}
	    }
	}
	out << "CELLS " << m << " " << m + n << std::endl;
	int l = 0;
	for (int i = 0; i < parameters_list.size(); i++) {
	    for (int j = 0; j < parameters_list[i].result_lines.size(); j++) {
		out << (parameters_list[i].result_lines)[j].size();
		for (int k = 0; k < (parameters_list[i].result_lines)[j].size(); k++) {
		    out << " " << l;
		    l++;
		}
		out << std::endl;
	    }
	}
	out << "CELL_TYPES " << m << std::endl;
	for (int i = 0; i < m; i++) {
	    out << 4 << std::endl;
	}

	return parameters_list;

	
    }

} //namespace cura


void InfillTestMain () {
    //cura::generateInfillTests();
    cura::generateInfillTests(" ");
}













  /*
    class InfillTest : public testing::TestWithParam<InfillTestParameters> {};

    INSTANTIATE_TEST_CASE_P(InfillTestcases, InfillTest, testing::ValuesIn(generateInfillTests()), [](testing::TestParamInfo<InfillTestParameters> info) { return info.param.name; });

    TEST_P(InfillTest, TestInfillSanity)
    {
        InfillTestParameters params = GetParam();
        ASSERT_TRUE(params.valid) << params.fail_reason;
        ASSERT_FALSE(params.result_polygons.empty() && params.result_lines.empty()) << "Infill should have been generated.";

#ifdef TEST_INFILL_SVG_OUTPUT
        writeTestcaseSVG(params);
#endif //TEST_INFILL_SVG_OUTPUT

        const double min_available_area = std::abs(params.outline_polygons.offset(-params.params.line_distance / 2).area());
        const double max_available_area = std::abs(params.outline_polygons.offset( params.params.line_distance / 2).area());
        const double min_expected_infill_area = (min_available_area * infill_line_width) / params.params.line_distance;
        const double max_expected_infill_area = (max_available_area * infill_line_width) / params.params.line_distance;

        const double out_infill_area = ((params.result_polygons.polygonLength() + params.result_lines.polyLineLength()) * infill_line_width) / getPatternMultiplier(params.params.pattern);

        ASSERT_GT((coord_t)max_available_area, (coord_t)out_infill_area) << "Infill area should allways be less than the total area available.";
        ASSERT_GT((coord_t)out_infill_area, (coord_t)min_expected_infill_area) << "Infill area should be greater than the minimum area expected to be covered.";
        ASSERT_LT((coord_t)out_infill_area, (coord_t)max_expected_infill_area) << "Infill area should be less than the maximum area to be covered.";

        const Polygons padded_shape_outline = params.outline_polygons.offset(infill_line_width / 2);
        ASSERT_EQ(padded_shape_outline.intersectionPolyLines(params.result_lines).polyLineLength(), params.result_lines.polyLineLength()) << "Infill (lines) should not be outside target polygon.";
        ASSERT_EQ(params.result_polygons.difference(padded_shape_outline).area(), 0) << "Infill (polys) should not be outside target polygon.";
    }
  */
