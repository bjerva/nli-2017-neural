for f in *.txt; do
cat "$f" | aspell list >  "$(basename "$f" .txt).err"
done
#
#aspell -a -c < 00017.err > 00017.err.sp
