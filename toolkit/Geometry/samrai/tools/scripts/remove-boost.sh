find . -name '*.[Ch]' -exec sed -i 's/boost::shared_ptr/std::shared_ptr/g' {} \;
find . -name '*.[Ch]' -exec sed -i 's/boost::\(.*\)_pointer_cast/std::\1_pointer_cast/g' {} \;
find . -name '*.[Ch]' -exec sed -i 's/BOOST_CAST/POINTER_CAST/g' {} \;
find . -name '*.[Ch]' -exec sed -i 's/boost::make_shared/std::make_shared/g' {} \;
find . -name '*.[Ch]' -exec sed -i '/boost\/make_shared/d' {} \;
find . -name '*.[Ch]' -exec sed -i '/boost\/shared_ptr/d' {} \;
