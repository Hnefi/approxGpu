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

    typedef boost::container::vector< std::pair<float,DqPtr> > VectorOfTablePairs;

    // get K/V pairs and sort by key (float that came from the 3-GHB average)
    VectorOfTablePairs sorted(myMap.begin(),myMap.end());
    std::sort(sorted.begin(),sorted.end(),&compPairs);

    int prev_size = sorted.size();
    float mult_fac = (float) max_size / (float) prev_size;

    // copy the first max_size-1 values
    FloatVector out(max_size);
    for(int i = 0;i < max_size && i < prev_size;i++) {
        DqPtr d = sorted[i].second;
        out[i] = d->at(0); // this is true because we averaged them down previously
    }

    // now calculate I' = I * (mult_fac) = I * (max_size / prev_size), for all the upper order values
    //  then average these values in with the previous I' values
    for(int i = max_size; i < prev_size; i++) {
        DqPtr d = sorted[i].second;
        float valToAvg = d->at(0);
        int i_prime = boost::math::iround( (float)i * mult_fac );
        assert(i_prime < max_size);
        out[i_prime] = (out[i_prime] + valToAvg) / 2.0;
    }

    return out;

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
