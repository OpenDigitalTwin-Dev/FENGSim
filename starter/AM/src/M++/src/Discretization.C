// file: Discretization.C
// author: Christian Wieners
// $Header: /public/M++/src/Discretization.C,v 1.28 2009-11-24 09:46:35 wieners Exp $

#include "Discretization.h"

void Discretization::fill (int m, int dim) {
    clean();
    if (name == "linear") {
        int dcell = 2;
        ReadConfig(Settings,"QuadratureCell",dcell);
        if ( (dcell>3) || (dcell<1) ) { Exit("Quadrature not implemented!"); }
        int dbnd = max(dcell-1,1);
        ReadConfig(Settings,"QuadratureBoundary",dbnd);
        if ( (dbnd>3) || (dbnd<1) ) { Exit("Quadrature not implemented!"); }
        char cell3D[128];
        char bnd3D[128];
        sprintf(cell3D,"Qhex%i",dcell*dcell*dcell);
        sprintf(bnd3D,"Qquad%i",dbnd*dbnd);

		S[0][0] = new P1tri(GetQuadrature("Qtri16"),
							GetQuadrature("Qint3"));
		S[1][0] = new P1quad(GetQuadrature("Qquad1"),
							 GetQuadrature("Qint1"));
		S[2][0] = new P1tet(GetQuadrature("Qtet1"),
							GetQuadrature("Qtri1"));
		S[4][0] = new P1hex(GetQuadrature(cell3D),
							GetQuadrature(bnd3D));
    } 
    else if (name == "CR_linear") {
		S[0][0] = new CR_P1tri(GetQuadrature("Qtri1"),
							   GetQuadrature("Qint1"));
    }
    else if (name == "serendipity") {
		S[0][0] = new P2tri(GetQuadrature("Qtri3"),
							GetQuadrature("Qint1"));
		S[1][0] = new P2quadSerendipity(GetQuadrature("Qquad9"),
										GetQuadrature("Qint2"));
		S[4][0] = new P2hexSerendipity(GetQuadrature("Qhex27"),
									   GetQuadrature("Qquad4"));
    } 
    else if (name == "quadratic") {
		S[0][0] = new P2tri(GetQuadrature("Qtri3"),
							GetQuadrature("Qint1"));
		S[1][0] = new P2quad(GetQuadrature("Qquad9"),
							 GetQuadrature("Qint2"));
		S[4][0] = new P2hex(GetQuadrature("Qhex27"),
							GetQuadrature("Qquad4"));
    } 
    else if (name == "quadsp") {
		S[4][0] = new P2hex_sp(GetQuadrature("Qhex27"),
							   GetQuadrature("Qquad4"));
    }
    else if (name == "cubic") {
		S[1][0] = new P3quad(GetQuadrature("Qquad16"),
							 GetQuadrature("Qint2"));
		S[4][0] = new P3hex(GetQuadrature("Qhex27"),
							GetQuadrature("Qquad4"));
    } 
    else if (name == "curl") {
        S[2][0] = new P1curl_tet(GetQuadrature("Qtet4"),
								 GetQuadrature("Qtri1"));
		S[4][0] = new P1curl_hex(GetQuadrature("Qhex8"),
								 GetQuadrature("Qquad1"));
    } 
    else if (name == "QuasiMagnetoStaticsMixed") {
      	S[2][0] = new P1tet(GetQuadrature("Qtet4"),
							GetQuadrature("Qtri1"));
        S[2][1] = new P1curl_tet(GetQuadrature("Qtet4"),
								 GetQuadrature("Qtri1"));
    }
    else if (name == "curl2") {
	S[4][0] = new P2curl_sp_hex(GetQuadrature("Qhex27"),
				    GetQuadrature("Qquad4"));
    } 
    else if (name == "TaylorHoodSerendipity_P2P1") {
	S[0][0] = new P1tri(GetQuadrature("Qtri7"));
	S[0][1] = new P2tri(GetQuadrature("Qtri7"));
	S[1][0] = new P1quad(GetQuadrature("Qquad9"));
	S[4][0] = new P1hex(GetQuadrature("Qhex27"),GetQuadrature("Qquad4"));
	S[1][1] = new P2quadSerendipity(GetQuadrature("Qquad9"));
	S[4][1] = new P2hexSerendipity(GetQuadrature("Qhex27"),
				       GetQuadrature("Qquad4"));
    }
    else if (name == "TaylorHoodQuadratic_P2P1") {
	S[0][0] = new P1tri(GetQuadrature("Qtri7"));
	S[0][1] = new P2tri(GetQuadrature("Qtri7"));
	S[1][0] = new P1quad(GetQuadrature("Qquad9"));
	S[4][0] = new P1hex(GetQuadrature("Qhex27"),GetQuadrature("Qquad4"));

	S[1][1] = new P2quad(GetQuadrature("Qquad9"));
	S[4][1] = new P2hex(GetQuadrature("Qhex27"),GetQuadrature("Qquad4"));
    }
    else if (name == "EqualOrderP1P1") {
        S[0][0] = new P1tri(GetQuadrature("Qtri16"),GetQuadrature("Qint5"));
	S[0][1] = new P1tri(GetQuadrature("Qtri16"),GetQuadrature("Qint5"));
	S[1][0] = new P1quad(GetQuadrature("Qquad4"));
	S[1][1] = new P1quad(GetQuadrature("Qquad4"));
	S[2][0] = new P1tet(GetQuadrature("Qtet1"),GetQuadrature("Qtri1"));
	S[2][1] = new P1tet(GetQuadrature("Qtet1"),GetQuadrature("Qtri1"));
	S[4][0] = new P1hex(GetQuadrature("Qhex8"));
	S[4][1] = new P1hex(GetQuadrature("Qhex8"));
    }
    else if (name == "EqualOrderP1P1PPP") {
        S[0][0] = new P1tri(GetQuadrature("Qtri16"),GetQuadrature("Qint5"));
	S[0][1] = new P1tri(GetQuadrature("Qtri16"),GetQuadrature("Qint5"));
	S[0][2] = new P1triPPP(GetQuadrature("Qtri16"),GetQuadrature("Qint5"));
	/*S[1][0] = new P1quad(GetQuadrature("Qquad4"));
	S[1][1] = new P1quad(GetQuadrature("Qquad4"));
	S[2][0] = new P1tet(GetQuadrature("Qtet1"),GetQuadrature("Qtri1"));
	S[2][1] = new P1tet(GetQuadrature("Qtet1"),GetQuadrature("Qtri1"));
	S[4][0] = new P1hex(GetQuadrature("Qhex8"));
	S[4][1] = new P1hex(GetQuadrature("Qhex8"));*/
    }
    else if (name == "EqualOrderP2P2") {
	S[0][0] = new P2tri(GetQuadrature("Qtri7"),
			    GetQuadrature("Qint1"));
	S[0][1] = new P2tri(GetQuadrature("Qtri7"),
			    GetQuadrature("Qint1"));
	S[1][0] = new P2quad(GetQuadrature("Qquad9"),
                             GetQuadrature("Qint2"));
	S[1][1] = new P2quad(GetQuadrature("Qquad9"),
                             GetQuadrature("Qint2"));
	S[4][0] = new P2hex(GetQuadrature("Qhex27"),
                            GetQuadrature("Qquad4"));
	S[4][1] = new P2hex(GetQuadrature("Qhex27"),
                            GetQuadrature("Qquad4"));
    }
    else if ( (name == "CosseratP1") 
              || (name == "CosseratP2P1") 
              || (name == "CosseratP2") ) {
        int dcell;
        if (name == "CosseratP1")        dcell = 2;
        else if (name == "CosseratP2P1") dcell = 3;
        else if (name == "CosseratP2")   dcell = 3;
        ReadConfig(Settings,"QuadratureCell",dcell);
        if ( (dcell>3) || (dcell<1) ) { Exit("Quadrature not implemented!"); }
        int dbnd = max(dcell-1,1);
        ReadConfig(Settings,"QuadratureBoundary",dbnd);
        if ( (dbnd>3) || (dbnd<1) ) { Exit("Quadrature not implemented!"); }
        char cell3D[128];
        char cell2D[128];
        char bnd3D[128];
        char bnd2D[128];
        sprintf(cell3D,"Qhex%i",dcell*dcell*dcell);
        sprintf(cell2D,"Qquad%i",dcell*dcell);
        sprintf(bnd3D,"Qquad%i",dbnd*dbnd);
        sprintf(bnd2D,"Qint%i",dbnd);
        if (name == "CosseratP1") {
            S[1][0] = new P1quad(GetQuadrature(cell2D),
                                 GetQuadrature(bnd2D));
            S[1][1] = new P1quad(GetQuadrature(cell2D),
                                 GetQuadrature(bnd2D));
            S[4][0] = new P1hex(GetQuadrature(cell3D),
                                GetQuadrature(bnd3D));
            S[4][1] = new P1hex(GetQuadrature(cell3D),
                                GetQuadrature(bnd3D));
        }
        else if (name == "CosseratP2P1") {
            S[1][0] = new P2quadSerendipity(GetQuadrature(cell2D),
                                            GetQuadrature(bnd2D));
            S[1][1] = new P1quad(GetQuadrature(cell2D),
                                 GetQuadrature(bnd2D));
            S[4][0] = new P2hexSerendipity(GetQuadrature(cell3D),
                                           GetQuadrature(bnd3D));
            S[4][1] = new P1hex(GetQuadrature(cell3D),
                                GetQuadrature(bnd3D));
        }
        else if (name == "CosseratP2") {
            S[1][0] = new P2quadSerendipity(GetQuadrature(cell2D),
                                            GetQuadrature(bnd2D));
            S[1][1] = new P2quadSerendipity(GetQuadrature(cell2D),
                                            GetQuadrature(bnd2D));
            S[4][0] = new P2hexSerendipity(GetQuadrature(cell3D),
                                           GetQuadrature(bnd3D));
            S[4][1] = new P2hexSerendipity(GetQuadrature(cell3D),
                                           GetQuadrature(bnd3D));
        }
    }
    else if (name == "THserendipity") {
	S[4][0] = new P2hexSerendipity(GetQuadrature("Qhex27"),
				       GetQuadrature("Qquad4"));
	S[4][1] = new P1hex(GetQuadrature("Qhex27"),GetQuadrature("Qquad4"));
    }
    else if (name == "THquadratic") {
	S[0][0] = new P2tri(GetQuadrature("Qtri7"),
			    GetQuadrature("Qint1"));
	S[0][1] = new P2tri(GetQuadrature("Qtri7"),
			    GetQuadrature("Qint1"));
	S[1][0] = new P2quad(GetQuadrature("Qquad9"),
                             GetQuadrature("Qint2"));
	S[1][1] = new P2quad(GetQuadrature("Qquad9"),
                             GetQuadrature("Qint2"));
	S[4][0] = new P2hex(GetQuadrature("Qhex27"),
                            GetQuadrature("Qquad4"));
	S[4][1] = new P2hex(GetQuadrature("Qhex27"),
                            GetQuadrature("Qquad4"));
    }
    else if (name == "cell") {
        S[0][0] = new P0(GetQuadrature("Qtri3"),
			 GetQuadrature("Qint1"));
    }
    else if (name == "RT0_P0") {
	S[0][0] = new P0(GetQuadrature("Qtri3"),
			 GetQuadrature("Qint1"));
	S[0][1] = new RT0tri(GetQuadrature("Qtri3"),
			     GetQuadrature("Qint1"));
	S[1][0] = new P0(GetQuadrature("Qquad4"),
			 GetQuadrature("Qint1"));
	S[1][1] = new RT0quad(GetQuadrature("Qquad4"),
			      GetQuadrature("Qint1"));
	S[4][0] = new P0(GetQuadrature("Qhex8"),
			 GetQuadrature("Qint1"));
	S[4][1] = new RT0hex(GetQuadrature("Qhex8"),
			     GetQuadrature("Qint1"));
/*
  S[0][0] = new P0(GetQuadrature("Qtri1"),
  GetQuadrature("Qint1"));
  S[0][1] = new RT_P1tri(GetQuadrature("Qtri1"),
  GetQuadrature("Qint1"));
*/
    }
    else if (name == "dGscalar") {
	S[0][0] = new P1tri(GetQuadrature("Qtri1"),
			    GetQuadrature("Qint1"));
	S[1][0] = new P1quad(GetQuadrature("Qquad4"),
			     GetQuadrature("Qint1"));
	S[2][0] = new P1tet(GetQuadrature("Qtet11"),
			    GetQuadrature("Qtri7"));
	S[4][0] = new P1hex(GetQuadrature("Qhex8"),
			    GetQuadrature("Qquad1"));
    } 
    else if (name =="dGscalar_P2"){
	S[0][0] = new P2tri(GetQuadrature("Qtri3"),
			    GetQuadrature("Qint2"));
	S[1][0] = new P2quad(GetQuadrature("Qquad4"),
			     GetQuadrature("Qint2"));
	S[2][0] = new P2tet(GetQuadrature("Qtet11"),
			    GetQuadrature("Qtri7"));
	S[4][0] = new P2hex(GetQuadrature("Qhex27"),
			    GetQuadrature("Qquad4"));
	}
    else if (name == "dGvector") {
	S[0][0] = new P1tri(GetQuadrature("Qtri1"),
			    GetQuadrature("Qint1"));
	S[1][0] = new P1quad(GetQuadrature("Qquad4"),
			     GetQuadrature("Qint1"));
	S[2][0] = new P1tet(GetQuadrature("Qtet11"),
			    GetQuadrature("Qtri7"));
	S[4][0] = new P1hex(GetQuadrature("Qhex8"),
			    GetQuadrature("Qquad1"));
    } 
    else Exit(name + " not implemented; file: Discretization.C\n");
}

void Discretization::fill (const Discretization& D, int m, int dim) {
    clean();
    if (name == "linear") {
	if (D.S[0][0] != 0) S[0][0] = new P1tri(D.S[0][0]->GetQuad());
	if (D.S[1][0] != 0) S[1][0] = new P1quad(D.S[1][0]->GetQuad());
	if (D.S[2][0] != 0) S[2][0] = new P1tet(D.S[2][0]->GetQuad());
	if (D.S[4][0] != 0) S[4][0] = new P1hex(D.S[4][0]->GetQuad());
    } 
    else if (name == "serendipity") {
	if (D.S[1][0]!=0) S[1][0]=new P2quadSerendipity(D.S[1][0]->GetQuad());
	if (D.S[4][0]!=0) S[4][0]=new P2hexSerendipity(D.S[4][0]->GetQuad());
    } 
    else if (name == "quadratic") {
	if (D.S[4][0] != 0) S[4][0] = new P2hex(D.S[4][0]->GetQuad());
    }
    else if (name == "quadsp") {
	if (D.S[4][0] != 0) S[4][0] = new P2hex_sp(D.S[4][0]->GetQuad());
    }
    else if (name == "cubic") {
	if (D.S[1][0] != 0) S[1][0] = new P3quad(D.S[1][0]->GetQuad());
	if (D.S[4][0] != 0) S[4][0] = new P3hex(D.S[4][0]->GetQuad());
    } 
    else if (name == "curl") {
	if (D.S[2][0] != 0) S[2][0] = new P1curl_tet(D.S[2][0]->GetQuad());
	if (D.S[4][0] != 0) S[4][0] = new P1curl_hex(D.S[4][0]->GetQuad());
    } 
    else if (name == "curl2") {
	if (D.S[4][0] != 0) S[4][0] = new P2curl_sp_hex(D.S[4][0]->GetQuad());
    } 
    else if (name == "EqualOrderP2P2") {
	S[0][0] = new P2tri(D.S[0][0]->GetQuad());
	S[0][1] = new P2tri(D.S[0][1]->GetQuad());
	S[1][0] = new P2quad(D.S[1][0]->GetQuad());
	S[1][1] = new P2quad(D.S[1][1]->GetQuad());
	S[4][0] = new P2hex(D.S[4][0]->GetQuad());
	S[4][1] = new P2hex(D.S[4][1]->GetQuad());
    }
    else if (name == "dGscalar") {
      S[0][0] = new P1tri(D.S[0][0]->GetQuad(),
			  D.S[0][0]->GetFaceQuad());
      S[1][0] = new P1quad(D.S[1][0]->GetQuad(),
			   D.S[1][0]->GetFaceQuad());
      S[2][0] = new P1tet(D.S[2][0]->GetQuad(),
			  D.S[2][0]->GetFaceQuad());
      S[4][0] = new P1hex(D.S[4][0]->GetQuad(),
			  D.S[4][0]->GetFaceQuad());
    } 
    else if (name =="dGscalar_P2"){
      S[0][0] = new P2tri(D.S[0][0]->GetQuad(),
			  D.S[0][0]->GetFaceQuad());
      S[1][0] = new P2quad(D.S[1][0]->GetQuad(),
			   D.S[1][0]->GetFaceQuad());
      S[2][0] = new P2tet(D.S[2][0]->GetQuad(),
			  D.S[2][0]->GetFaceQuad());
      S[4][0] = new P2hex(D.S[4][0]->GetQuad(),
			  D.S[4][0]->GetFaceQuad());
    }
    else if (name == "dGvector") {
      S[0][0] = new P1tri(D.S[0][0]->GetQuad(),
			  D.S[0][0]->GetFaceQuad());
      S[1][0] = new P1quad(D.S[1][0]->GetQuad(),
			   D.S[1][0]->GetFaceQuad());
      S[2][0] = new P1tet(D.S[2][0]->GetQuad(),
			  D.S[2][0]->GetFaceQuad());
      S[4][0] = new P1hex(D.S[4][0]->GetQuad(),
			  D.S[4][0]->GetFaceQuad());
    } 
    else Exit(name + " not implemented; file: Discretization.C\n");
}
