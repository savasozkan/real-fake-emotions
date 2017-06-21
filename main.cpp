
#include "vector"
#include <unistd.h>
#include "fstream"
#include "omp.h"

#include <glob.h>

#include <iostream>
#include "cnpy.h"

#include "CSVM.h"
#include "operators.h"

int str2int(string sval)
{
	int ival;
	istringstream ss(sval);
	ss >> ival;

	return ival;
}

void l2norm(vector<float>& feat)
{
	double nr = 0.0;
	int ln = feat.size();
	for(int i=0; i<ln; i++)
		nr += feat[i]*feat[i];

	nr = sqrt(nr);
	if(nr > 1e-13)
	{
		for(int i=0; i<ln; i++)
			feat[i] /= nr;
	}
}

vector<float> element_mean(const vector<vector<float> >& velem, int nt)
{
	int ncount = velem.size();
	vector<float> vsum(nt, 0.0);
	for(int i=0; i<ncount; i++)
	{
		vsum += velem[i];
	}

	return vsum / float(ncount) ;
}

vector<float> element_var(const vector<vector<float> >& velem, const vector<float>& vmean, int nt)
{
	int ncount = velem.size();
	vector<float> vvar(nt, 0.0);
	for(int i=0; i<ncount; i++)
	{
		vvar += (velem[i]-vmean)*(velem[i]-vmean);
	}

	for(int i=0; i<nt; i++)
		vvar[i] = sqrt(vvar[i]);

	return vvar / float(ncount) ;
}

vector<float> element_max(const vector<vector<float> >& velem, int nt)
{
	int ncount = velem.size();
	vector<float> vsum(nt, 0.0);
	for(int i=0; i<ncount; i++)
	{
		for(int j=0; j<velem[i].size(); j++)
			vsum[j] = max(vsum[j], velem[i][j]);
	}

	return vsum;
}

float sign(const float& v)
{
	return (v >= 0) ? 1 : -1;
}

float power(const float& v)
{
	return pow(abs(v), 1.5);
}

void power_norm(vector<float>& vcode)
{
	for(int i=0; i<int(vcode.size()); i++)
		vcode[i] = sign(vcode[i])*power(vcode[i]);
}


vector<float> dot_prod(const vector<float>& v1, const vector<float>& v2, int nt)
{
	vector<float> vp(nt*(nt+1)/2, 0.0);
	int cntr = 0;
	for(int i1=0; i1<nt; i1++)
		for(int i2=i1; i2<nt; i2++, cntr++)
			vp[cntr] = v1[i1]*v2[i2];

	return vp;
}

vector<float> covrepr(const vector<vector<float> >& vfeat1, const vector<vector<float> >& vfeat2, int nt)
{
	int ncount = vfeat1.size();
	vector<float> vm1 = element_mean(vfeat1, nt);
	vector<float> vm2 = element_mean(vfeat2, nt);
	vector<float> vc(nt*(nt+1)/2, 0.0);
	for(int i=0; i<ncount; i++)
	{
		vector<float> vdiff1 = vfeat1[i]-vm1;
		vector<float> vdiff2 = vfeat2[i]-vm2;
		vc += dot_prod(vdiff1, vdiff2, nt);
	}

	vc /= float(ncount);

	return vc;
}

vector<float> covrepr(const vector<vector<float> >& vfeat, int nt)
{
	int ncount = vfeat.size();
	vector<float> vm = element_mean(vfeat, nt);
	vector<float> vc(nt*(nt+1)/2, 0.0);
	for(int i=0; i<ncount; i++)
	{
		vector<float> vdiff = vfeat[i]-vm;
		//l2norm(vdiff);
		vc += dot_prod(vdiff, vdiff, nt);
	}

	vc /= float(ncount);

	return vc;
}

vector<float> maxrepr(const vector<vector<float> >& vfeat, int nt)
{
	return element_max(vfeat, nt);
}

void read_npydata(const string& pthfile, vector<vector<float> >& vdatai)
{
	vdatai.clear();
	cnpy::NpyArray arr = cnpy::npy_load(pthfile);
	float* data = reinterpret_cast<float*>(arr.data);

	int nelem = arr.shape[0];
	int ndim  = arr.shape[1];

	for(int i=0; i<nelem; i++)
	{
		vector<float> vdata(data+i*ndim, data+(i+1)*ndim);
		vdatai.push_back(vdata);
	}
}

vector<float> facetoupper(const vector<float>& vcode, int nc, int nw, int nh)
{
	vector<float> vupper(nc*nh*nw/8, 0);
	int cntr = 0;
	for(int c=0; c<nc; c+=4)
		for(int h=0; h<nh/2; h++)
			for(int w=0; w<nw; w++, cntr++)
			{
				double val = 0;
				for(int ic=0; ic<4; ic++)
					val = vcode[(c+ic)*nh*nw + h*nw + w];

				vupper[cntr] = val/double(4);
			}

	return vupper;
}

vector<float> facetolower(const vector<float>& vcode, int nc, int nw, int nh)
{
	vector<float> vlower(nc*nh*nw/8, 0);
	int cntr = 0;
	for(int c=0; c<nc; c+=4)
		for(int h=nh/2; h<nh; h++)
			for(int w=0; w<nw; w++, cntr++)
			{
				double val = 0;
				for(int ic=0; ic<4; ic++)
					val = vcode[(c+ic)*nh*nw + h*nw + w];

				vlower[cntr] = val/double(4);
			}

	return vlower;
}

const int ndim = 128;

void train(string emotion)
{
	cout<<endl<<"#Training Phase"<<endl<<endl;
	string filepath = "features/"+emotion+"/train/";

	vector<vector<float> > vfeat;
	vector<int> vlabel;
	vector<path> vtrainlist = path(filepath+"*.npy").Glob();
	for(int f=0; f<int(vtrainlist.size()); f++)
	{
		int l = str2int( string(vtrainlist[f].BaseName().RawName().RawName().RawName().RawName().Extension()) );

		vector<vector<float> > vdatai;
		read_npydata(string(vtrainlist[f]), vdatai);
		vector<float> ft = covrepr(vdatai, ndim);
		power_norm(ft); l2norm(ft);

		vfeat.push_back(ft);
		vlabel.push_back(l);
	}

	{
		CSVM svm(C_SVC, LINEAR, 1, 0, 0.0, 0.1, 1000, 10);
		svm.train(vfeat, vlabel);
		svm.write("svmmodels/"+emotion+".svm");
	}

	{
		CSVM svm;
		svm.read("svmmodels/"+emotion+".svm");

		float acc =0;
		for(int f=0; f<vfeat.size(); f++)
		{
			int pr = svm.predict(vfeat[f]);
			acc += (vlabel[f] == pr) ? 1 : 0;
		}

		cout<<"Training accuracy: "<<acc/double(vfeat.size())<<endl;
	}
}

void validate(string emotion)
{
	cout<<endl<<"#Validation Phase"<<endl<<endl;

	string filepath = "features/"+emotion+"/val/";

	vector<vector<float> > vtestfeat;
	vector<path> vtestlist = path(filepath+"*.npy").Glob();
	for(int f=0; f<int(vtestlist.size()); f++)
	{
		vector<vector<float> > vdatai;
		read_npydata(string(vtestlist[f]), vdatai);
		vector<float> ft = covrepr(vdatai, ndim);
		power_norm(ft); l2norm(ft); //power_norm(ft);

		vtestfeat.push_back(ft);
	}

	{
		CSVM svm;
		svm.read("svmmodels/"+emotion+".svm");

		for(int f=0; f<vtestfeat.size(); f++)
		{
			int pr = svm.predict(vtestfeat[f]);
			string stype = (pr == 1) ? "real" : "fake";
			cout<<vtestlist[f].BaseName()<<" "<<pr<< "("<<stype<<")"<<endl;
		}
	}
}

void test(string emotion)
{
	string filepath = "features/"+emotion+"/test/";
}


int main(int argc, char** argv)
{
	train("anger");
	validate("anger");
}

