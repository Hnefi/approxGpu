#include "genBuckets.h"

using std::size_t;
using std::cout;
using std::endl;
using namespace boost::algorithm;

bool compPairs( const std::pair<float,DqPtr> one, const std::pair<float,DqPtr> two) {
    if (one.first <= two.first) return true;
    else return false;
}

    FloatVector
LVAData::GenerateUniqueValueSet(unsigned int max_size)
{
    // read of all threads/files
    slaveReader.processInput();
    slaveReader.reduceAverages();
    // print to hashmap file
    PrintTables();

    typedef boost::container::vector< std::pair<float,DqPtr> > VectorOfTablePairs;

    // get K/V pairs and sort by key (float that came from the 3-GHB average)
    VectorOfTablePairs sorted(myMap.begin(),myMap.end());
    std::sort(sorted.begin(),sorted.end(),&compPairs);

    int prev_size = sorted.size();
    float mult_fac = (float) max_size / (float) prev_size;
    /*
    cout << "mult_fac is: " << mult_fac << endl;;

    cout << "----sorted unique k/v pairs----" << endl;
    for(int i = 0;i<prev_size;i++)
        cout << sorted[i].second->at(0) << endl;
        */

    // copy the first max_size-1 values
    FloatVector approxSet(max_size);
    for(int i = 0;i < max_size && i < prev_size;i++) {
        DqPtr d = sorted[i].second;
        approxSet[i] = d->at(0); // this is true because we averaged them down previously
    }

    /*
    cout << "----approx_set values----" << endl;
    for(int i = 0;i<max_size;i++)
        cout << approxSet[i] << endl;
        */

    // now calculate I' = I * (mult_fac) = I * (max_size / prev_size), for all the upper order values
    //  then average these values in with the previous I' values
    for(int i = max_size; i < prev_size; i++) {
        DqPtr d = sorted[i].second;
        float valToAvg = d->at(0);
        int i_prime = boost::math::iround( (float)i * mult_fac );
        cout << "i: " << i << ", i_prime: " << i_prime << endl;
        //int i_prime = ( i % max_size );
        assert(i_prime < max_size);
        approxSet[i_prime] = (approxSet[i_prime] + valToAvg) / 2.0;
    }

    return approxSet;

    /*
    // make a big vector of all the inputs that exit and sort.
    FloatVector input_vals;
    GHBHashToValueMap::iterator it = myMap.begin();
    for(; it != myMap.end(); it++) {
    std::pair<float,DqPtr> v = *it;
    DqPtr this_dq = v.second;
    for(int i = 0; i < this_dq->size() ; i++)
    input_vals.push_back(this_dq->at(i));
    }
    boost::sort(input_vals);
    FloatSet unique_vals(input_vals.begin(),input_vals.end());
    FloatVector unique_vect_to_compress(unique_vals.begin(),unique_vals.end());

    //setup random functions
    _BASE_GENERATOR generator(static_cast<unsigned int>(std::time(0)));

    boost::container::vector<int> indices_to_remove;
    int numToRemove = (unique_vect_to_compress.size() - max_size);

    typedef boost::variate_generator<_BASE_GENERATOR&, distribution_type> _INT_GEN_TYPE;
    _INT_GEN_TYPE die_generator(generator,distribution_type(0,unique_vect_to_compress.size()-1));

    // generate random indices to remove
    for(int i = 0; i <= numToRemove;i++)
    indices_to_remove.push_back(die_generator());

    // now remove all these elements from the big vector
    for(size_t i = 0; i < indices_to_remove.size();i++) {
    std::swap(unique_vect_to_compress[indices_to_remove[i]],unique_vect_to_compress.back());
    unique_vect_to_compress.pop_back();
    }
     */
}

int main(int argc, char* argv[])
{
    std::string inputTexFile;
    if(argc != 2) {
        std::cout << "Just give me a text file already...." << std::endl;
        exit(EXIT_FAILURE);
    } else {
        // turn argv[1] into a string
        inputTexFile.assign(argv[1]);
    }

    LVAData tex_array_maker(inputTexFile);
    //tex_array_maker.PrintTables();
    // create a big set of values that represents our texture memory
    FloatVector tex_arr = tex_array_maker.GenerateUniqueValueSet(65535); // this is max texture 1D for a 780GTX

    std::ofstream dumpFile;
    dumpFile.open("trainingSet.txt");
    // print this tex arr to file.
    for(int i = 0; i < tex_arr.size(); i++) {
        dumpFile << tex_arr[i] << std::endl;

    }
    dumpFile.close();
    return 0;
}
