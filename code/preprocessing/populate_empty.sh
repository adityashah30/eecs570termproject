sed -i -e 's/^,/0,/' -e':a;s/,,/,0,/;ta' -e 's/,\s$/,0/' ../../data/links.csv
sed -i -e 's/^,/0,/' -e':a;s/,,/,0,/;ta' -e 's/,\s$/,0/' ../../data/movies.csv
sed -i -e 's/^,/0,/' -e':a;s/,,/,0,/;ta' -e 's/,\s$/,0/' ../../data/ratings.csv
sed -i -e 's/^,/0,/' -e':a;s/,,/,0,/;ta' -e 's/,\s$/,0/' ../../data/tags.csv
