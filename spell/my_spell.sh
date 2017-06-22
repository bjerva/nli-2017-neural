for f in *.err; do
cat "$f" | aspell -a -c >  "$(basename "$f" .err).cor"
done
#
#aspell -a -c < 00017.err > 00017.err.sp
