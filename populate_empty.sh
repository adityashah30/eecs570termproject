sed -i -e 's/^,/0,/' -e':a;s/,,/,0,/;ta' -e 's/,\s$/,0/' {file}
