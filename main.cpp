
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

float sign(const float& v)
{
	return (v >= 0) ? 1 : -1;
}

float power(const float& v)
{
	return pow(abs(v), 0.5);
}

void power_norm(vector<float>& vcode)
{
	for(int i=0; i<int(vcode.size()); i++)
		vcode[i] = sign(vcode[i])*power(vcode[i]);
}

void read_npydata(const string& pthfile, vector<float>& ft)
{
	ft.clear();
	cnpy::NpyArray arr = cnpy::npy_load(pthfile);
	float* data = reinterpret_cast<float*>(arr.data);

	int ndim  = arr.shape[1];

	vector<float> vdata(data, data+ndim);
	ft = vdata;
}

void train(string emotion)
{
	cout<<endl<<"#Training Phase"<<endl<<endl;
	string filepath = "features/"+emotion+"/train/";

	vector<vector<float> > vfeat;
	vector<int> vlabel;
	vector<path> vtrainlist = path(filepath+"*.npy").Glob();
	for(int f=0; f<int(vtrainlist.size()); f++)
	{
		int l = str2int( string(vtrainlist[f].BaseName().RawName().RawName().RawName().Extension()) );

		vector<float> ft;
		read_npydata(string(vtrainlist[f]), ft);
		power_norm(ft);

		vfeat.push_back(ft);
		vlabel.push_back(l);
	}

	{
		CSVM svm(C_SVC, LINEAR, 1, 0, 0.0, 1, 1000, 1);
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
		vector<float> ft;
		read_npydata(string(vtestlist[f]), ft);
		power_norm(ft);

		vtestfeat.push_back(ft);
	}

	{
		CSVM svm;
		svm.read("svmmodels/"+emotion+".svm");

		for(int f=0; f<vtestfeat.size(); f++)
		{
			int pr = svm.predict(vtestfeat[f]);
			string stype = (pr == 1) ? "real" : "fake";
			cout<<vtestlist[f].BaseName()<<" "<<pr<< " ("<<stype<<")"<<endl;
		}
	}
}

void test(string emotion)
{

	cout<<endl<<"#Test Phase"<<endl<<endl;

	string filepath = "features/"+emotion+"/test/";

	vector<vector<float> > vtestfeat;
	vector<path> vtestlist = path(filepath+"*.npy").Glob();
	for(int f=0; f<int(vtestlist.size()); f++)
	{
		vector<float> ft;
		read_npydata(string(vtestlist[f]), ft);
		power_norm(ft);

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


int main(int argc, char** argv)
{
//	train("anger");
//	train("happiness");
//	train("surprise");
//	train("disgust");
//	train("contentment");
//	train("sadness");

	validate("anger");
	validate("happiness");
	validate("surprise");
	validate("disgust");
	validate("contentment");
	validate("sadness");

	test("anger");
	test("happiness");
	test("surprise");
	test("disgust");
	test("contentment");
	test("sadness");
}

