tr -d '\r' < ../../data/links.csv > ../../data/links.csv.out
tr -d '\r' < ../../data/movies.csv > ../../data/movies.csv.out
tr -d '\r' < ../../data/ratings.csv > ../../data/ratings.csv.out
tr -d '\r' < ../../data/tags.csv > ../../data/tags.csv.out
mv ../../data/links.csv.out ../../data/links.csv
mv ../../data/movies.csv.out ../../data/movies.csv
mv ../../data/ratings.csv.out ../../data/ratings.csv
mv ../../data/tags.csv.out ../../data/tags.csv
sed -i -e 's/^,/0,/' -e':a;s/,,/,0,/;ta' -e 's/,$/,0/' ../../data/links.csv
sed -i -e 's/^,/0,/' -e':a;s/,,/,0,/;ta' -e 's/,$/,0/' ../../data/movies.csv
sed -i -e 's/^,/0,/' -e':a;s/,,/,0,/;ta' -e 's/,$/,0/' ../../data/ratings.csv
sed -i -e 's/^,/0,/' -e':a;s/,,/,0,/;ta' -e 's/,$/,0/' ../../data/tags.csv
